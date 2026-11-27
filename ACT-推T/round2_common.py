"""Shared causal state features and rollout code for the second experiment."""
import json, math, os
from collections import deque
from pathlib import Path
import numpy as np
import torch
from act_model import ACT

ROOT=Path(__file__).resolve().parent
R2=ROOT/'round2'
DT=.05
FEATURE_DIM=21

def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)

def velocities(raw,previous=None):
    raw=np.asarray(raw,dtype=np.float32)
    if previous is None:return np.zeros(raw.shape[:-1]+(5,),np.float32)
    previous=np.asarray(previous,dtype=np.float32)
    a=np.arctan2(raw[...,4],raw[...,5]);b=np.arctan2(previous[...,4],previous[...,5])
    omega=np.arctan2(np.sin(a-b),np.cos(a-b))/DT
    return np.concatenate([(raw[...,2:4]-previous[...,2:4])/DT,omega[...,None],(raw[...,:2]-previous[...,:2])/DT],-1).astype(np.float32)

def episode_velocities(raw):
    v=np.zeros((len(raw),5),np.float32)
    v[1:]=velocities(raw[1:],raw[:-1])
    return v

def features(raw,vel):
    raw=np.asarray(raw,dtype=np.float32);vel=np.asarray(vel,dtype=np.float32)
    delta=raw[...,6:8]-raw[...,2:4]
    local=raw[...,:2]-raw[...,2:4]
    s,c=raw[...,4],raw[...,5]
    local=np.stack([c*local[...,0]+s*local[...,1],-s*local[...,0]+c*local[...,1]],-1)
    # Continuous relative angle (cos delta, sin delta), with no angle wrapping.
    rel=np.stack([raw[...,9]*c+raw[...,8]*s,raw[...,8]*c-raw[...,9]*s],-1)
    return np.concatenate([raw,delta,rel,local,vel],-1).astype(np.float32)

def perturb(raw,vel,rng):
    raw=raw.copy();vel=vel.copy()
    raw[...,:4]+=rng.normal(0,.001,raw[...,:4].shape)
    a=np.arctan2(raw[...,4],raw[...,5])+rng.normal(0,.01,raw.shape[:-1])
    raw[...,4]=np.sin(a);raw[...,5]=np.cos(a)
    # Velocity noise is independent; differentiating position noise would create
    # artificial high frequency motion absent from the causal observation.
    vel[...,:2]+=rng.normal(0,.002,vel[...,:2].shape)
    vel[...,2]+=rng.normal(0,.02,vel[...,2].shape)
    vel[...,3:]+=rng.normal(0,.002,vel[...,3:].shape)
    return raw,vel

class Policy:
    def __init__(self,path,n_action_steps=1,temporal_coeff=None):
        self.path=str(path);self.checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        cfg=self.checkpoint['config'];self.cfg=cfg;self.stats=self.checkpoint['stats']
        self.enhanced=cfg.get('feature_version')=='relative_velocity_v1'
        self.vision=cfg.get('vision',False)
        self.model=ACT(vision=self.vision,state_dim=FEATURE_DIM if self.enhanced else 10,dropout=cfg.get('dropout',.1)).cuda().eval()
        self.model.load_state_dict(self.checkpoint['model'])
        self.steps=n_action_steps;self.temporal=temporal_coeff
        self.mean=np.asarray(self.stats['mean'],np.float32);self.std=np.asarray(self.stats['std'],np.float32)
        self.am=np.asarray(self.stats['action_mean'],np.float32);self.astd=np.asarray(self.stats['action_std'],np.float32)
        self.reset()
    def reset(self):
        self.queue=deque();self.previous=None;self.history=[];self.t=0
    @torch.inference_mode()
    def action(self,raw,image=None):
        vel=velocities(raw,self.previous);self.previous=raw.copy()
        if self.temporal is not None or not self.queue:
            feat=features(raw,vel) if self.enhanced else raw
            state=torch.tensor((feat-self.mean)/self.std,dtype=torch.float32,device='cuda')[None]
            im=torch.tensor(image.transpose(2,0,1).copy(),device='cuda')[None] if self.vision else None
            with torch.autocast('cuda',dtype=torch.bfloat16):plan=self.model(state,im)[0].float().cpu().numpy()
            plan=plan*self.astd+self.am
            if self.temporal is None:self.queue.extend(plan[:self.steps])
            else:self.history.append((self.t,plan))
        if self.temporal is None:action=self.queue.popleft()
        else:
            self.history=[(t,p) for t,p in self.history if self.t-t<len(p)]
            values=np.stack([p[self.t-t] for t,p in self.history])
            weights=np.exp(-self.temporal*np.array([self.t-t for t,p in self.history]))
            action=np.average(values,axis=0,weights=weights)
        self.t+=1
        return np.clip(action,-1,1).astype(np.float32)

def classify(trace):
    m=trace['metrics'];raw=trace['states']
    maximum=float(m[:,0].max());last=m[-1]
    if maximum>=.95:return 'success'
    if maximum>=.90 and maximum-last[0]>.05:return 'overshoot_after_near_alignment'
    tail=raw[-80:]
    ee_motion=float(np.linalg.norm(np.diff(tail[:,:2],axis=0),axis=1).sum())
    block_motion=float(np.linalg.norm(np.diff(tail[:,2:4],axis=0),axis=1).sum())
    if ee_motion<.01 and block_motion<.003:return 'stalled'
    if maximum>=.75 and last[3]>.05:return 'fine_angular_alignment'
    if maximum>=.75:return 'fine_translation_or_reposition'
    return 'failed_to_reach_alignment_region'

def rollout(env,policy,seed,trace_path=None,horizon=600):
    env.reset_task(seed);policy.reset();states=[];actions=[];metrics=[]
    for _ in range(horizon):
        if (ROOT/'STOP_REQUESTED').exists():raise InterruptedError('Stop requested')
        raw=env.state_vector()
        im=env.sim.render(128,128,camera_name='birdview')[::-1].copy() if policy.vision else None
        action=policy.action(raw,im)
        states.append(raw);actions.append(action);env.move(action)
        m=env.metrics();metrics.append([m[k] for k in ['coverage','iou','distance','angle','z','tilt']])
        if m['coverage']>=.95 or m['tilt']>.2 or m['z']<.79 or np.max(np.abs(env.xy()))>.35:break
    trace=dict(states=np.asarray(states),actions=np.asarray(actions),metrics=np.asarray(metrics))
    if trace_path:atomic_npz(trace_path,seed=np.int64(seed),**trace)
    m=trace['metrics'];idx=int(np.argmax(m[:,0]))
    return dict(seed=int(seed),success=bool(m[idx,0]>=.95 and m[:,5].max()<.2),max_coverage=float(m[idx,0]),final_coverage=float(m[-1,0]),distance=float(m[-1,2]),angle=float(m[-1,3]),steps=len(actions),failure_type=classify(trace))

def evaluate(policy,seeds,out=None,traces=False):
    from push_t_env import make_env
    env=make_env(policy.vision);rows=[]
    try:
        for seed in seeds:
            rows.append(rollout(env,policy,seed,Path(out)/f'trace_{seed}.npz' if traces else None))
            if out:save_json(Path(out)/'progress.json',dict(completed=len(rows),target=len(seeds),last=rows[-1]))
    finally:env.close()
    n=len(rows);p=float(np.mean([r['success'] for r in rows]));z=1.96
    center=(p+z*z/(2*n))/(1+z*z/n);margin=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    result=dict(success_rate=p,successes=sum(r['success'] for r in rows),n=n,success_ci95=[center-margin,center+margin],mean_max_coverage=float(np.mean([r['max_coverage'] for r in rows])),mean_distance=float(np.mean([r['distance'] for r in rows])),mean_angle=float(np.mean([r['angle'] for r in rows])),episodes=rows)
    if out:save_json(Path(out)/'summary.json',result)
    return result
