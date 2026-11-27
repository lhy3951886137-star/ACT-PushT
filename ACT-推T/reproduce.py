"""Four-model entry point with archived evaluation settings and fresh output paths."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def train(key, card, name, dry):
    module=card['training_module']
    source=ROOT/Path(*module.split('.')).with_suffix('.py')
    flags={}
    for n in ast.walk(ast.parse(source.read_text(encoding='utf-8'))):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant):
            flag=n.args[0].value
            if isinstance(flag,str) and flag.startswith('--'):
                flags[flag[2:].replace('-','_')]=(flag,{kw.arg:ast.literal_eval(kw.value) for kw in n.keywords if kw.arg=='action'})
    cfg=json.loads((ROOT/card['config']).read_text(encoding='utf-8'))
    cfg.update(card.get('training_overrides',{})); cfg['name']=name or cfg['name']+'_retrain'
    if (ROOT/'act_v2/runs'/cfg['name']).exists():
        raise SystemExit('Training destination already exists; choose a fresh --name: '+cfg['name'])
    args=[sys.executable,'-m',module]
    for k,v in cfg.items():
        if k not in flags or v is None: continue
        flag,options=flags[k]
        if options.get('action')=='store_true':
            if v: args.append(flag)
        else:
            args.append(flag)
            args.extend(str(x) for x in (v if isinstance(v,list) else [int(v) if isinstance(v,bool) else v]))
    print(subprocess.list2cmdline(args),flush=True)
    if not dry: subprocess.run(args,cwd=ROOT,check=True)


def evaluate(card,split,count,out):
    import torch
    if not torch.cuda.is_available(): raise SystemExit('These archived policy runners require a CUDA-enabled PyTorch installation.')
    torch.set_num_threads(1)
    spec=card['splits'].get(split)
    if spec is None: raise SystemExit('Available splits: '+', '.join(card['splits']))
    start,n=spec
    if count is not None and not 1<=count<=n: raise SystemExit('--count must be within the archived split size')
    seeds=list(range(start,start+(count or n)))
    path=ROOT/card['checkpoint']
    if not path.exists(): raise SystemExit('Download weights first: python download_assets.py --preset primary')
    kwargs={'checkpoint_path':path,'weight_kind':'ema',**card['inference']}
    family=card['family']
    if family=='full-vision':
        from act_v2.full_random_vision import FullVisionACTPolicy as Policy, evaluate_full_vision as evaluate
    elif family=='full-state':
        from act_v2.full_random_state import FullRandomStateACTPolicy as Policy, evaluate_full_state as evaluate
    elif family=='small-state':
        from act_v2.inference import StateACTPolicy as Policy, evaluate_state as evaluate
    else:
        from act_v2.inference import VisionACTPolicy as Policy, evaluate_vision as evaluate
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output=Path(out) if out else ROOT/'reproduced'/family/f'{split}-{stamp}'
    if output.exists() and any(output.iterdir()): raise SystemExit('Use a fresh --out directory; existing results would be cached by the archived evaluators.')
    extra={'zero_images':True} if split=='zero' else {}
    result=evaluate(Policy(**kwargs),seeds,out_dir=output,**extra)
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2),flush=True)
    print('Results:',output)


def main():
    cards=json.loads((ROOT/'models.json').read_text(encoding='utf-8'))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['eval','train']); p.add_argument('model',choices=list(cards))
    p.add_argument('--split',default='blind'); p.add_argument('--count',type=int)
    p.add_argument('--out'); p.add_argument('--name'); p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(); os.chdir(ROOT)
    if args.action=='train': train(args.model,cards[args.model],args.name,args.dry_run)
    else: evaluate(cards[args.model],args.split,args.count,args.out)


if __name__=='__main__': main()
