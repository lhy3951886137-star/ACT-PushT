"""Freeze the oracle; preserve confirmation failures; archive abandoned control."""
import hashlib,shutil,math
import numpy as np
from vision_io import ROOT,S2,save_json,read_json

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare():
    dest=S2/'archive/state_oracle';dest.mkdir(parents=True,exist_ok=True)
    source=ROOT/'round4/runs/precision_diffusion_100k/best.pt'
    expected='21eee80e5e93869cafd7c9549935cf4f6fe0c3bfb665332ce058ce45658670e0'
    assert sha(source)==expected
    files=['round4/runs/precision_diffusion_100k/best.pt','round4/runs/precision_diffusion_100k/config.json','round4/runs/precision_diffusion_100k/stats.json','round4/state_gate.json','round4/frozen_precision_model.json','round4/protocol.json','precision_env.py','push_t_env.py','precision_collect.py','round4/assets/t_precision.xml','assets/t_block.xml','round4_runtime.py','round3_models.py','round3_data.py']
    hashes={}
    for rel in files:
        src=ROOT/rel;dst=dest/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(src,dst)
        assert sha(src)==sha(dst);hashes[rel]=sha(src)
    rows=[]
    summary=read_json(ROOT/'round4/confirmation/summary.json')
    for row in summary['episodes']:
        if row['success']:continue
        seed=row['seed'];folder=dest/'failures';folder.mkdir(exist_ok=True)
        for filename in [f'trace_{seed}.npz',f'seed_{seed}.json']:shutil.copy2(ROOT/'round4/confirmation'/filename,folder/filename)
        with np.load(folder/f'trace_{seed}.npz') as d:
            s=d['states'];m=d['metrics'];peak=int(m[:,0].argmax());a=d['actions']
            initial=s[0];angle=float(np.degrees(np.arctan2(initial[4],initial[5])));distance=float(np.linalg.norm(initial[2:4]-initial[6:8]))
            row=dict(row,initial_block_xy=initial[2:4].tolist(),initial_goal_xy=initial[6:8].tolist(),initial_angle_deg=angle,initial_distance_mm=distance*1000,peak_step=peak+1,tail_block_travel_mm=float(np.linalg.norm(np.diff(s[-100:,2:4],axis=0),axis=1).sum()*1000),tail_ee_travel_mm=float(np.linalg.norm(np.diff(s[-100:,:2],axis=0),axis=1).sum()*1000),tail_mean_action_norm=float(np.linalg.norm(a[-100:],axis=1).mean()),ended_out_of_bounds=bool(abs(s[-1,:2]).max()>.345),near_angle_boundary=abs(angle)>math.degrees(.12)*.9,near_distance_boundary=distance<.0495 or distance>.0855)
            rows.append(row)
    save_json(dest/'freeze.json',dict(checkpoint_step=80000,n_action_steps=4,inference_steps=20,checkpoint_sha256=expected,source_sha256=hashes,confirmation_success=.92,failures=rows,interpretation='Four cases are descriptive diagnostics, not causal evidence. Preserve exact oracle inference (20 DDIM); vision uses a separate 16-step policy.'))
    old=ROOT/'round4/runs/legacy_diffusion_100k'
    save_json(S2/'archive/legacy_control.json',dict(status='abandoned_by_user',last_status=read_json(old/'status.json'),artifacts=str(old),resume_allowed=False,reason='User instructed archival without repairing or completing the 67.8k control.'))
    save_json(S2/'curriculum_gate.json',dict(stage=2,course='unchanged small displacement/angle',required_confirmation_success=.85,confirmation_scenes=50,stage3_allowed=False,stage3A=dict(demos=250,angle_deg=60,workspace_fraction=.65,small_course_fraction=.28),stage3B=dict(demos=[350,500],angle_deg=180,state_first=True,failure_patch_demos=[50,80]),note='Stage 3 remains gated until the frozen Vision model passes independent confirmation.'))
    return rows

if __name__=='__main__':print(prepare())
