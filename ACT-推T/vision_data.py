"""Verified physical replay -> predecoded uint8 dual-view arrays (no codecs)."""
import argparse,hashlib,os,time
from pathlib import Path
import numpy as np
from PIL import Image
from vision_io import ROOT,S2,atomic,save_json,read_json,stopped
from vision_env import make_env,observe,CAMERAS,CAMERA_CONFIG
from precision_collect import rollout,qualified

DATA=S2/'data'

def npz(path,**arrays):
    def writer(p):
        with p.open('wb') as f:np.savez_compressed(f,**arrays)
    atomic(path,writer)

def export_episode(env,data,row,index):
    n=len(data['actions']);prefix=f'episode_{index:04d}';rgb=DATA/(prefix+'_rgb.npy');metadata=DATA/(prefix+'.npz')
    env.reset_task(int(data['seed']));maximum=0.;started=time.monotonic()
    def render(tmp):
        nonlocal maximum
        output=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(n,2,128,128,3))
        for t,a in enumerate(data['actions']):
            stopped();maximum=max(maximum,float(np.max(np.abs(env.state_vector()-data['states'][t]))))
            if maximum>1e-5:raise RuntimeError(f'Physics replay mismatch: {maximum}')
            output[t]=observe(env)['images']
            if index==0 and t in (0,n//2,n-1):
                for c,name in enumerate(CAMERAS):
                    folder=S2/'previews';folder.mkdir(exist_ok=True)
                    Image.fromarray(output[t,c]).save(folder/f'{name}_{t:04d}.png')
            env.move(a)
            if t%100==0:save_json(DATA/'status.json',dict(status='rendering',pid=os.getpid(),episode=index+1,target_episodes=150,frame=t+1,frames=n,seed=int(data['seed']),max_replay_error=maximum))
        output.flush();del output
    atomic(rgb,render)
    maximum=max(maximum,float(np.max(np.abs(env.state_vector()-data['terminal_state']))))
    if maximum>1e-5 or not qualified(env.metrics()):raise RuntimeError('Terminal replay failed')
    # Privileged annotations stay outside the model dataset. Coverage is only a sampler weight.
    npz(metadata,robot_xy=data['states'][:,:2],actions=data['actions'],coverage=data['coverage'])
    result=dict(index=index,seed=int(data['seed']),frames=n,partition=row['partition'],rgb=rgb.name,metadata=metadata.name,source=row.get('source'),max_replay_error=maximum,terminal_metrics=env.metrics(),seconds=time.monotonic()-started,rgb_bytes=rgb.stat().st_size)
    save_json(DATA/(prefix+'.json'),result);print(result,flush=True)
    return result

def collect(limit=150):
    DATA.mkdir(parents=True,exist_ok=True)
    if limit==150 and (DATA/'complete.json').exists():return read_json(DATA/'complete.json')
    base=read_json(ROOT/'round4/precision_data/complete.json')['episodes'];rows=[];env=make_env()
    try:
        for i,row in enumerate(base[:min(100,limit)]):
            stopped();record=DATA/f'episode_{i:04d}.json'
            if record.exists():rows.append(read_json(record));continue
            path=ROOT/'round4/precision_data'/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
            with np.load(path) as d:data={k:d[k] for k in d.files}
            rows.append(export_episode(env,data,dict(row,source=str(path)),i))
        attempt=0
        while len(rows)<limit:
            stopped();seed=4000000+attempt;attempt+=1;trial=DATA/f'attempt_{seed}.json'
            if trial.exists():
                r=read_json(trial)
                if r['success']:rows.append(read_json(DATA/f"episode_{r['index']:04d}.json"))
                continue
            save_json(DATA/'status.json',dict(status='collecting_new_expert',pid=os.getpid(),completed=len(rows),target_episodes=150,seed=seed))
            data,r=rollout(env,seed)
            if r['success']:
                i=len(rows);raw=DATA/f'expert_{seed}.npz';npz(raw,**data)
                # New validation examples remain disjoint from both fixed evaluation sets.
                partition='validation' if (i-100)%7==0 else 'train'
                rows.append(export_episode(env,data,dict(partition=partition,source=str(raw)),i));r['index']=i
            save_json(trial,r)
            if attempt>=1000:raise RuntimeError('Too many failed expert trials')
    finally:env.close()
    result=dict(episodes=rows,total_frames=sum(r['frames'] for r in rows),image_bytes=sum(r['rgb_bytes'] for r in rows),cameras=CAMERAS,camera_config=CAMERA_CONFIG,policy_inputs=['robot_xy[2]','RGB birdview','RGB robot0_eye_in_hand'],privileged_policy_inputs=[],physics='unchanged precision_env',replay_tolerance=1e-5)
    if limit==150:save_json(DATA/'complete.json',result);save_json(DATA/'status.json',dict(status='complete',pid=os.getpid(),episodes=len(rows),total_frames=result['total_frames']))
    return result

class Sampler:
    def __init__(self,manifest,stats=None,validation=False,seed=25):
        rows=[r for r in manifest['episodes'] if (r['partition']=='validation')==validation]
        if not rows:raise ValueError('No trajectories in partition')
        self.episodes=[];self.rng=np.random.default_rng(seed)
        for r in rows:
            with np.load(DATA/r['metadata']) as d:meta={k:d[k] for k in d.files}
            assert set(meta)=={'robot_xy','actions','coverage'} and meta['robot_xy'].shape[1]==2
            self.episodes.append(dict(**meta,images=np.load(DATA/r['rgb'],mmap_mode='r')))
        self.lengths=np.asarray([len(d['actions']) for d in self.episodes]);self.weights=self.lengths/self.lengths.sum()
        self.fine=[np.flatnonzero(d['coverage']>.90) for d in self.episodes]
        if stats is None:
            xy=np.concatenate([d['robot_xy'] for d in self.episodes]);lo=xy.min(0);hi=xy.max(0)
            stats=dict(robot_center=((lo+hi)/2).tolist(),robot_scale=np.maximum((hi-lo)/2,.01).tolist(),action_center=[0.,0.],action_scale=[1.,1.])
        self.stats=stats;self.validation=validation
    def sample(self,batch):
        robot=[];images=[];actions=[];valid=[]
        for i in self.rng.choice(len(self.episodes),size=batch,p=self.weights):
            e=self.episodes[i];n=len(e['actions']);fine=self.fine[i]
            t=int(self.rng.choice(fine)) if not self.validation and len(fine) and self.rng.random()<.5 else int(self.rng.integers(n))
            history=[max(0,t-1),t];future=t+np.arange(16)
            robot.append(e['robot_xy'][history]);images.append(e['images'][history]);actions.append(e['actions'][np.minimum(future,n-1)]);valid.append(future<n)
        xy=(np.asarray(robot,np.float32)-np.asarray(self.stats['robot_center'],np.float32))/np.asarray(self.stats['robot_scale'],np.float32)
        return xy,np.asarray(images,np.uint8),np.asarray(actions,np.float32),np.asarray(valid,bool)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=150);a=p.parse_args();collect(a.limit)
