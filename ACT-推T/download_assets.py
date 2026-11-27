"""Download GitHub Release assets anonymously when public, verify SHA256, and restore paths."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8*1024*1024): h.update(b)
    return h.hexdigest()


def get_token():
    token=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if token: return token
    try:
        r=subprocess.run(['gh','auth','token'],text=True,capture_output=True,check=False)
        if r.returncode==0 and r.stdout.strip(): return r.stdout.strip()
    except FileNotFoundError: pass
    try:
        env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
        r=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,env=env)
        values=dict(s.split('=',1) for s in r.stdout.splitlines() if '=' in s)
        if values.get('password'): return values['password']
    except FileNotFoundError: pass
    raise SystemExit('GitHub authentication required for a private repository or API rate limits. Sign in with gh auth login, or provide GH_TOKEN / GITHUB_TOKEN.')


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        result=super().redirect_request(request,fp,code,msg,headers,newurl)
        if result and urllib.parse.urlsplit(newurl).hostname!=urllib.parse.urlsplit(request.full_url).hostname:
            result.remove_header('Authorization')
        return result


def fetch_release(opener,manifest):
    url=f"https://api.github.com/repos/{manifest['repository']}/releases/tags/{manifest['tag']}"
    headers={'Accept':'application/vnd.github+json','User-Agent':'ACT-PushT-reproduction'}
    token=None
    try:
        with opener.open(urllib.request.Request(url,headers=headers),timeout=90) as response:
            return json.load(response),token
    except urllib.error.HTTPError as exc:
        if exc.code not in (401,403,404): raise
    token=get_token()
    headers['Authorization']='Bearer '+token
    with opener.open(urllib.request.Request(url,headers=headers),timeout=90) as response:
        return json.load(response),token


def asset_request(asset,token):
    headers={'Accept':'application/octet-stream','User-Agent':'ACT-PushT-reproduction'}
    if token is None:
        return urllib.request.Request(asset['browser_download_url'],headers=headers)
    headers['Authorization']='Bearer '+token
    return urllib.request.Request(asset['url']+'?download=1',headers=headers)


def select(manifest,preset):
    items=manifest['assets']
    if preset=='all': return items
    if preset=='primary': return [a for a in items if a.get('primary')]
    if preset=='weights': return [a for a in items if a['kind']=='file']
    model=preset.removeprefix('train-')
    cards=json.loads((ROOT/'models.json').read_text(encoding='utf-8'))
    card=cards[model]
    categories=set(card['data_categories'])
    warmstart=card.get('training_overrides',{}).get('init_from')
    return [a for a in items if a['category'] in categories or (a.get('path')==card['checkpoint']) or (warmstart is not None and a.get('path')==warmstart)]


def unpack(path,asset):
    import zstandard
    expected=set(asset['files']); found=set()
    with path.open('rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as stream:
        with tarfile.open(fileobj=stream,mode='r|') as archive:
            for member in archive:
                rel=PurePosixPath(member.name)
                if rel.is_absolute() or '..' in rel.parts or member.name not in expected or not member.isfile():
                    raise RuntimeError('Unsafe or unexpected archive entry: '+member.name)
                archive.extract(member,path=ROOT,filter='data')
                found.add(member.name)
    if found!=expected: raise RuntimeError('Archive file inventory differs from manifest: '+asset['name'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',choices=['primary','weights','all','train-full-vision','train-full-state','train-small-vision','train-small-state'],default='primary')
    p.add_argument('--list',action='store_true'); p.add_argument('--keep-cache',action='store_true')
    p.add_argument('--verify-local',action='store_true',help='Verify restored files against archived byte-for-byte source hashes, without network access')
    p.add_argument('--asset-dir',type=Path,help='Use previously downloaded local assets, without network access')
    args=p.parse_args(); manifest=json.loads((ROOT/'assets-manifest.json').read_text(encoding='utf-8'))
    assets=select(manifest,args.preset)
    for a in assets: print(f"{a['name']}  {a['size']/1e6:.1f} MB  [{a['category']}]")
    print(f'Total: {len(assets)} assets, {sum(a["size"] for a in assets)/1e9:.3f} GB')
    if args.list: return
    if args.verify_local:
        hashes=json.loads((ROOT/'dataset-file-checksums.json').read_text(encoding='utf-8'))
        files={}
        for a in assets:
            if a['kind']=='file': files[a['path']]={'sha256':a['sha256'],'size':a['size']}
            else:
                for item in hashes[a['name']]: files[item['path']]=item
        for path,item in files.items():
            target=ROOT/path
            if not target.exists() or target.stat().st_size!=item['size'] or sha(target)!=item['sha256']:
                raise SystemExit('Missing or altered local file: '+path)
        print(f'All {len(files)} restored files match their original SHA256 hashes.')
        return
    cache=ROOT/'.asset-cache'; cache.mkdir(exist_ok=True)
    completed_path=cache/'extracted.json'
    completed=json.loads(completed_path.read_text()) if completed_path.exists() else {}
    opener=urllib.request.build_opener(SafeRedirect())
    token=None; remote={}
    for a in assets:
        if a['kind']=='file':
            dest=ROOT/a['path']
            if dest.exists() and sha(dest)==a['sha256']:
                print('Verified existing',a['path']); continue
        elif completed.get(a['name'])==a['sha256'] and all((ROOT/f).exists() for f in a['files']):
            print('Already extracted',a['name']); continue
        local=(args.asset_dir/a['name']) if args.asset_dir else cache/a['name']
        if not local.exists() or local.stat().st_size!=a['size'] or sha(local)!=a['sha256']:
            if args.asset_dir: raise SystemExit('Missing/corrupt local asset: '+str(local))
            if not remote:
                release,token=fetch_release(opener,manifest)
                remote={item['name']:item for item in release['assets']}
            asset=remote.get(a['name'])
            if not asset: raise SystemExit('Release asset missing: '+a['name'])
            tmp=local.with_suffix(local.suffix+'.partial')
            last_error=None
            for attempt in range(3):
                try:
                    request=asset_request(asset,token)
                    with opener.open(request,timeout=180) as response,tmp.open('wb') as target:
                        shutil.copyfileobj(response,target,length=4*1024*1024)
                    if tmp.stat().st_size!=a['size'] or sha(tmp)!=a['sha256']: raise RuntimeError('Downloaded checksum/size mismatch: '+a['name'])
                    tmp.replace(local); last_error=None; break
                except Exception as exc:
                    last_error=exc
                    if attempt<2: time.sleep(2)
            if last_error: raise last_error
        print('SHA256 OK',a['name'],flush=True)
        if a['kind']=='file':
            dest=ROOT/a['path']; dest.parent.mkdir(parents=True,exist_ok=True)
            if args.asset_dir or args.keep_cache: shutil.copy2(local,dest)
            else: local.replace(dest)
        else:
            unpack(local,a)
            completed[a['name']]=a['sha256']
            completed_path.write_text(json.dumps(completed,indent=2),encoding='utf-8')
            if not args.keep_cache and not args.asset_dir: local.unlink()
        print('Restored',a.get('path',a['name']),flush=True)


if __name__=='__main__': main()
