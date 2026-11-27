"""Visual-only camera configuration; precision dynamics remain unchanged."""
import numpy as np
from precision_env import make_env as precision_env

CAMERAS=('birdview','robot0_eye_in_hand')
CAMERA_CONFIG=dict(birdview=dict(pos=[-.04,0,1.65],fovy=30.),robot0_eye_in_hand=dict(fovy=95.))

class GoalHighlighter:
    """Registry for visual goal indication; never supplied as a policy feature."""
    def __init__(self,env):self.env=env;self.registry={'push_t':('goal_cap','goal_stem')}
    def activate(self,name='push_t'):
        active=self.registry[name]
        for names in self.registry.values():
            for geom in names:
                gid=self.env.sim.model.geom_name2id(geom)
                self.env.sim.model.geom_rgba[gid,3]=.55 if geom in active else .15

def make_env():
    env=precision_env(True)
    # hard_reset=False keeps these model changes across deterministic resets.
    for name,cfg in CAMERA_CONFIG.items():
        cid=env.sim.model.camera_name2id(name)
        if 'pos' in cfg:env.sim.model.cam_pos[cid]=cfg['pos']
        env.sim.model.cam_fovy[cid]=cfg['fovy']
    env.goal_highlighter=GoalHighlighter(env);env.goal_highlighter.activate()
    env.sim.forward()
    return env

def observe(env):
    """Only robot proprioception and pixels cross the policy boundary."""
    images=np.stack([env.sim.render(width=128,height=128,camera_name=c)[::-1].copy() for c in CAMERAS])
    return dict(robot_xy=np.asarray(env.xy(),np.float32),images=images)
