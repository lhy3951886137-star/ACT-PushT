"""New precision demonstrations: 98% coverage, <1.5mm, <1.5deg, five holds."""
import argparse,json,time,hashlib,os
from collections import deque
import numpy as np
from shapely.geometry import Polygon,Point,LineString
from precision_env import make_env,VERTICES,STEP,R4,rotation
from round2_common import ROOT,save_json,atomic_npz

def qualified(m):
    return m['coverage']>=.98 and m['distance']<.0015 and m['angle']<np.deg2rad(1.5) and m['tilt']<.2

class Expert:
    def __init__(self):self.contact=None;self.remaining=0;self.stage='approach';self.route=deque()
    def plan(self,env):
        p,a=env.pose();R=rotation(a);xy=env.xy();delta=env.goal[:2]-p[:2]
        angle=np.arctan2(np.sin(env.goal[2]-a),np.cos(env.goal[2]-a));desired=np.r_[delta/.06,angle*.9]
        com=env.sim.model.body_ipos[env.cube_body_id,:2].copy()
        clearance=Polygon(VERTICES@R.T+p[:2]).buffer(.0118,join_style=2);candidates=[]
        for v1,v2 in zip(VERTICES,np.roll(VERTICES,-1,axis=0)):
            tangent=(v2-v1)/np.linalg.norm(v2-v1);normal=np.array([-tangent[1],tangent[0]])
            for f in np.linspace(.05,.95,19):
                local=v1+(v2-v1)*f;world=p[:2]+R@local;n=R@normal
                if clearance.contains(Point(world-n*.014)):continue
                arm=R@(local-com);twist=np.r_[n,np.cross(arm,n)/.045]
                score=float(desired@twist/(np.linalg.norm(twist)+1e-9))-.002*np.linalg.norm(world-xy)
                candidates.append((score,local,normal))
        _,self.contact,self.normal=max(candidates,key=lambda x:x[0])
        self.remaining=30;self.stage='approach';self.route=deque();self.planned_pose=np.r_[p[:2],a]
    def route_target(self,xy,target,obstacle,center):
        if not self.route:
            safety=obstacle.buffer(-.0002)
            if not LineString([xy,target]).intersects(safety):return target
            nodes=[xy,target,*np.asarray(obstacle.exterior.coords)[:-1]];dist=[float('inf')]*len(nodes);dist[0]=0;previous={};visited=set()
            for _ in nodes:
                u=min((i for i in range(len(nodes)) if i not in visited),key=lambda i:dist[i])
                if not np.isfinite(dist[u]) or u==1:break
                visited.add(u)
                for v in range(1,len(nodes)):
                    if v==u or v in visited:continue
                    if LineString([nodes[u],nodes[v]]).intersects(safety):continue
                    cost=dist[u]+np.linalg.norm(np.asarray(nodes[u])-nodes[v])
                    if cost<dist[v]:dist[v]=cost;previous[v]=u
            if 1 in previous:
                route=[1];v=1
                while previous[v]!=0:v=previous[v];route.append(v)
                self.route=deque(np.asarray(nodes[i]) for i in reversed(route))
            else:
                radial=xy-center
                return xy+radial/(np.linalg.norm(radial)+1e-9)*.01
        while len(self.route)>1 and np.linalg.norm(xy-self.route[0])<.002:self.route.popleft()
        return self.route[0]
    def action(self,env):
        if self.contact is None or self.remaining<=0:self.plan(env)
        p,a=env.pose();R=rotation(a);xy=env.xy();normal=R@self.normal;contact=p[:2]+R@self.contact
        if self.stage=='approach' and (np.linalg.norm(p[:2]-self.planned_pose[:2])>.002 or abs(a-self.planned_pose[2])>.02):
            self.plan(env);normal=R@self.normal;contact=p[:2]+R@self.contact
        if self.stage=='approach':
            approach=contact-normal*.014
            if np.linalg.norm(xy-approach)<.002:self.stage='push';self.route=deque()
            else:
                obstacle=Polygon(VERTICES@R.T+p[:2]).buffer(.011,join_style=2)
                target=self.route_target(xy,approach,obstacle,p[:2])
                return np.asarray(np.clip((target-xy)/STEP*.9,-1,1),np.float32)
        gap=float((contact-xy)@normal)
        if gap>.010:self.remaining=max(self.remaining,15)
        else:self.remaining-=1
        delta=env.goal[:2]-p[:2];angle=abs(np.arctan2(np.sin(env.goal[2]-a),np.cos(env.goal[2]-a)))
        signed_angle=np.arctan2(np.sin(env.goal[2]-a),np.cos(env.goal[2]-a))
        arm=R@(self.contact-env.sim.model.body_ipos[env.cube_body_id,:2]);twist=np.r_[normal,np.cross(arm,normal)/.045]
        if np.r_[delta/.06,signed_angle*.9]@twist<0:
            self.remaining=0
            return np.asarray(-normal*.4,np.float32)
        increment=float(np.clip(np.linalg.norm(delta)*.2+angle*.004,.00008,.002))
        friction_compensation=.12*.21*9.81/300
        target=contact-normal*.008+normal*(friction_compensation+increment)
        displacement=target-xy;along=float(displacement@normal)
        command=(along*normal+.8*(displacement-along*normal))/STEP
        cap=.5 if env.metrics()['coverage']>.90 else .8
        return np.asarray(np.clip(command,-cap,cap),np.float32)

def rollout(env,seed):
    env.reset_task(seed);expert=Expert();states=[];actions=[];metrics=[];coverage=[];holds=0
    for t in range(2000):
        if (ROOT/'STOP_REQUESTED').exists():raise InterruptedError('Stop requested')
        m=env.metrics()
        if qualified(m):
            if holds>=5:break
            a=np.zeros(2,np.float32);holds+=1
        else:holds=0;a=expert.action(env)
        states.append(env.state_vector());coverage.append(m['coverage']);actions.append(a.copy());env.move(a);m=env.metrics()
        metrics.append([m[k] for k in ['coverage','iou','distance','angle','z','tilt']])
        if m['tilt']>.2 or m['z']<.79 or abs(env.xy()).max()>.35:break
    success=bool(holds>=5 and qualified(env.metrics()) and max(m[5] for m in metrics)<.2)
    data=dict(seed=np.int64(seed),states=np.asarray(states,np.float32),actions=np.asarray(actions,np.float32),coverage=np.asarray(coverage,np.float32),metrics=np.asarray(metrics),terminal_state=env.state_vector(),terminal_metrics=np.asarray([env.metrics()[k] for k in ['coverage','iou','distance','angle','z','tilt']]))
    row=dict(seed=seed,success=success,frames=len(actions),holds=holds,final=env.metrics())
    return data,row

def replay(env,data):
    env.reset_task(int(data['seed']));error=0.
    for raw,a in zip(data['states'],data['actions']):error=max(error,float(abs(env.state_vector()-raw).max()));env.move(a)
    return dict(seed=int(data['seed']),max_state_error=error,passed=bool(error<=1e-5 and qualified(env.metrics())),final=env.metrics())

def collect(target=100,max_attempts=1000):
    out=R4/'precision_data';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return json.loads((out/'complete.json').read_text())
    existing={}
    for p in out.glob('episode_*.npz'):
        with np.load(p) as d:existing[int(d['seed'])]=p
    rows=[json.loads(x) for x in (out/'attempts.jsonl').read_text().splitlines()] if (out/'attempts.jsonl').exists() else []
    checks=json.loads((out/'replay_audit.json').read_text()) if (out/'replay_audit.json').exists() else []
    env=make_env(False);accepted=len(existing);started=time.monotonic()
    save_json(out/'status.json',dict(status='collecting',pid=os.getpid(),accepted=accepted,target=target))
    try:
        for i in range(len(rows),max_attempts):
            if accepted>=target:break
            seed=2000000+i
            if seed in existing:
                row=dict(seed=seed,success=True,recovered_committed_file=True)
            else:
                data,row=rollout(env,seed)
                if row['success']:
                    if accepted<5:
                        check=replay(env,data);checks.append(check);save_json(out/'replay_audit.json',checks)
                        if not check['passed']:raise RuntimeError(f'Precision replay failed: {check}')
                    path=out/f'episode_{accepted:04d}.npz';atomic_npz(path,**data);existing[seed]=path;accepted+=1
            row.update(accepted=accepted,target=target,attempt=i+1,seconds=time.monotonic()-started)
            with (out/'attempts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            save_json(out/'status.json',dict(status='collecting',pid=os.getpid(),**row));print(json.dumps(row),flush=True)
    except InterruptedError:
        save_json(out/'status.json',dict(status='stopped',accepted=accepted,target=target));return None
    finally:env.close()
    if accepted<target:raise RuntimeError(f'Precision expert accepted {accepted}/{target} within {max_attempts} attempts; threshold not relaxed')
    paths=sorted(out.glob('episode_*.npz'));order=np.random.default_rng(20261125).permutation(len(paths));val=set(order[:14].tolist());episodes=[]
    for i,p in enumerate(paths):
        with np.load(p) as d:seed=int(d['seed']);n=len(d['actions']);terminal=d['terminal_metrics'].tolist()
        episodes.append(dict(file=p.name,seed=seed,frames=n,partition='validation' if i in val else 'train',terminal_metrics=terminal,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    result=dict(episodes=episodes,target=target,replay_checks=checks,environment='precision_v1',terminal_coverage=.98,terminal_distance_m=.0015,terminal_angle_deg=1.5,hold_steps=5)
    save_json(out/'complete.json',result);save_json(out/'status.json',dict(status='complete',accepted=accepted,target=target));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');a=p.parse_args()
    if a.probe:
        e=make_env(False)
        for seed in range(2000000,2000003):
            d,r=rollout(e,seed)
            if r['success']:r['replay']=replay(e,d)
            atomic_npz(R4/f'probe_{seed}.npz',**d);print(json.dumps(r),flush=True)
        e.close()
    else:collect()
