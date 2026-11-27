"""Native robosuite Panda Push-T. Policy controls XY increments at 20 Hz."""
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'glfw' if os.name == 'nt' else 'egl')
import numpy as np
from shapely.geometry import Polygon
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.controllers import load_composite_controller_config

ROOT = Path(__file__).resolve().parent
VERTICES = np.array([[-.015,-.06],[.015,-.06],[.015,.03],[.06,.03],[.06,.06],[-.06,.06],[-.06,.03],[-.015,.03]])
STEP = .01
HEIGHT = .825

class CylinderPusher(GripperModel):
    def __init__(self, idn=0):
        super().__init__(str(ROOT/'assets/pusher.xml'), idn=idn)
    @property
    def init_qpos(self):
        return None
    def format_action(self, action):
        return np.zeros(0)

GRIPPER_MAPPING['CylinderPusher'] = CylinderPusher

def rotation(a):
    return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])

class PushT(Lift):
    def _load_model(self):
        super()._load_model()
        arena = self.model.mujoco_arena
        arena.set_camera('birdview', pos=[0,0,1.65], quat=[1,0,0,0], camera_attribs={'fovy':'48'})
        # Goal is visual only; it cannot exert contact forces.
        import xml.etree.ElementTree as ET
        goal = ET.SubElement(arena.worldbody, 'body', name='goal', pos='0 0 0.801')
        for name, size, pos in [('cap','.06 .015 .0005','0 .045 0'),('stem','.015 .045 .0005','0 -.015 0')]:
            ET.SubElement(goal,'geom',name='goal_'+name,type='box',size=size,pos=pos,rgba='.1 .85 .2 .55',contype='0',conaffinity='0',group='1')
        self.cube = MujocoXMLObject(str(ROOT/'assets/t_block.xml'), name='tblock', obj_type='all', duplicate_collision_geoms=True)
        self.placement_initializer = UniformRandomSampler(name='TPlacement',mujoco_objects=self.cube,x_range=[-.03,.03],y_range=[-.03,.03],rotation=0,ensure_object_boundary_in_range=False,reference_pos=self.table_offset,z_offset=.002,rng=self.rng)
        self.model = ManipulationTask(arena, self.model.mujoco_robots, [self.cube])

    def reward(self, action=None):
        return 0.

    def _check_success(self):
        return False if not hasattr(self,'cube_body_id') else self.metrics()['coverage'] >= .95

    def pose(self):
        pos = self.sim.data.body_xpos[self.cube_body_id].copy()
        mat = self.sim.data.body_xmat[self.cube_body_id].reshape(3,3)
        return pos, float(np.arctan2(mat[1,0],mat[0,0]))

    def xy(self):
        return self.sim.data.site_xpos[self.robots[0].eef_site_id['right']][:2].copy()

    def state_vector(self):
        p,a = self.pose()
        return np.r_[self.xy(),p[:2],np.sin(a),np.cos(a),self.goal[:2],np.sin(self.goal[2]),np.cos(self.goal[2])].astype(np.float32)

    def metrics(self):
        p,a=self.pose()
        goal=getattr(self,'goal',np.zeros(3))
        poly=Polygon(VERTICES@rotation(a).T+p[:2])
        target=Polygon(VERTICES@rotation(goal[2]).T+goal[:2])
        inter=poly.intersection(target).area
        mat=self.sim.data.body_xmat[self.cube_body_id].reshape(3,3)
        return dict(coverage=float(inter/target.area),iou=float(inter/poly.union(target).area),distance=float(np.linalg.norm(p[:2]-goal[:2])),angle=float(abs(np.arctan2(np.sin(a-goal[2]),np.cos(a-goal[2])))),z=float(p[2]),tilt=float(np.arccos(np.clip(mat[2,2],-1,1))))

    def move(self, action):
        action=np.clip(np.asarray(action,dtype=float),-1,1)
        site=self.sim.data.site_xpos[self.robots[0].eef_site_id['right']]
        from scipy.spatial.transform import Rotation
        mat=self.sim.data.site_xmat[self.robots[0].eef_site_id['right']].reshape(3,3)
        target=getattr(self,'fixed_orientation',mat)
        angular=Rotation.from_matrix(target@mat.T).as_rotvec()/.5
        cmd=np.r_[action*STEP/.05,np.clip((HEIGHT-site[2])/.05,-1,1),np.clip(angular,-1,1)]
        return self.step(cmd)

    def reset_task(self, seed, difficulty='translation'):
        self.rng=np.random.default_rng(seed)
        self.placement_initializer.rng=self.rng
        self.goal=np.zeros(3)
        self.reset()
        self.fixed_orientation=self.sim.data.site_xmat[self.robots[0].eef_site_id['right']].reshape(3,3).copy()
        rng=np.random.default_rng(seed)
        self.goal=np.r_[rng.uniform(-.035,.035,2),0.]
        gid=self.sim.model.body_name2id('goal')
        self.sim.model.body_pos[gid,:2]=self.goal[:2]
        self.sim.model.body_quat[gid]=[1,0,0,0]
        # Warm-up is excluded from the demonstration; block is placed afterwards.
        start=self.goal[:2]+[-.17,-.02]
        for _ in range(100):
            self.move(np.clip((start-self.xy())/STEP,-1,1))
        direction=np.array([1.,rng.uniform(-.12,.12)])
        direction/=np.linalg.norm(direction)
        distance=rng.uniform(.045,.09)
        angle=rng.uniform(-.12,.12) if difficulty=='translation' else rng.uniform(-.45,.45)
        p=self.goal[:2]-direction*distance
        self.sim.data.set_joint_qpos(self.cube.joints[0],np.r_[p,.817,np.cos(angle/2),0,0,np.sin(angle/2)])
        self.sim.data.qvel[:]=0
        self.sim.forward()
        for _ in range(8): self.move(np.zeros(2))
        self.timestep=0
        self.done=False
        return self.state_vector()

def make_env(images=False):
    cfg=load_composite_controller_config(controller='BASIC')
    cfg['body_parts']['right']={'type':'OSC_POSE','input_max':1,'input_min':-1,'output_max':[.05,.05,.05,.5,.5,.5],'output_min':[-.05,-.05,-.05,-.5,-.5,-.5],'kp':150,'damping_ratio':1,'impedance_mode':'fixed','kp_limits':[0,300],'damping_ratio_limits':[0,10],'position_limits':None,'orientation_limits':None,'uncouple_pos_ori':True,'input_type':'delta','input_ref_frame':'world','interpolation':None,'ramp_ratio':.2}
    cfg['body_parts']['right']['gripper']={'type':'GRIP'}
    cfg['body_parts']={'right':cfg['body_parts']['right']}
    return PushT(robots='Panda',gripper_types='CylinderPusher',controller_configs=cfg,initialization_noise=None,has_renderer=False,has_offscreen_renderer=images,use_camera_obs=False,use_object_obs=False,camera_names='birdview',camera_heights=128,camera_widths=128,control_freq=20,horizon=600,ignore_done=True,hard_reset=False,seed=0,table_friction=(.45,.005,.0001))

if __name__=='__main__':
    import json,time
    e=make_env(images=True)
    t=time.monotonic()
    e.reset_task(1)
    from PIL import Image
    Image.fromarray(e.sim.render(128,128,camera_name='birdview')[::-1]).save(ROOT/'environment_smoke.png')
    print(json.dumps({'state':e.state_vector().tolist(),'metrics':e.metrics(),'action_dim':e.action_dim,'seconds':time.monotonic()-t}),flush=True)
    e.close()
