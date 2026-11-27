"""Versioned 15cm Push-T with an explicit 4mm policy action scale."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon
from push_t_env import PushT,ROOT,VERTICES as OLD_VERTICES,rotation,HEIGHT
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.controllers import load_composite_controller_config

R4=ROOT/'round4'
SCALE=1.25
VERTICES=OLD_VERTICES*SCALE
STEP=.004
OSC_SCALE=.02

def asset():
    path=R4/'assets/t_precision.xml';path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        tree=ET.parse(ROOT/'assets/t_block.xml')
        for geom in tree.findall('.//geom'):
            for key in ['size','pos']:
                values=np.fromstring(geom.get(key),sep=' ');values[:2]*=SCALE;geom.set(key,' '.join(map(str,values)))
            geom.set('friction','.12 .005 .0001');geom.set('solref','.03 1.2');geom.set('solimp','.9 .95 .001')
        site=tree.find('.//site[@name="horizontal_radius_site"]');site.set('pos',f'{.085*SCALE} 0 0')
        tree.write(path,encoding='unicode')
    return path

class PrecisionPushT(PushT):
    def _load_model(self):
        super()._load_model();arena=self.model.mujoco_arena
        for name in ['goal_cap','goal_stem']:
            geom=arena.worldbody.find(f'.//geom[@name="{name}"]')
            for key in ['size','pos']:
                a=np.fromstring(geom.get(key),sep=' ');a[:2]*=SCALE;geom.set(key,' '.join(map(str,a)))
        self.cube=MujocoXMLObject(str(asset()),name='tblock',obj_type='all',duplicate_collision_geoms=True)
        self.placement_initializer=UniformRandomSampler(name='TPlacement',mujoco_objects=self.cube,x_range=[-.03,.03],y_range=[-.03,.03],rotation=0,ensure_object_boundary_in_range=False,reference_pos=self.table_offset,z_offset=.002,rng=self.rng)
        self.model=ManipulationTask(arena,self.model.mujoco_robots,[self.cube])
    def move(self,action):
        a=np.clip(np.asarray(action,float),-1,1);site=self.sim.data.site_xpos[self.robots[0].eef_site_id['right']]
        mat=self.sim.data.site_xmat[self.robots[0].eef_site_id['right']].reshape(3,3)
        angular=Rotation.from_matrix(getattr(self,'fixed_orientation',mat)@mat.T).as_rotvec()/.5
        return self.step(np.r_[a*STEP/OSC_SCALE,np.clip((HEIGHT-site[2])/OSC_SCALE,-1,1),np.clip(angular,-1,1)])
    def metrics(self):
        p,a=self.pose();goal=getattr(self,'goal',np.zeros(3))
        polygon=Polygon(VERTICES@rotation(a).T+p[:2]);target=Polygon(VERTICES@rotation(goal[2]).T+goal[:2]);inter=polygon.intersection(target).area
        mat=self.sim.data.body_xmat[self.cube_body_id].reshape(3,3)
        return dict(coverage=float(inter/target.area),iou=float(inter/polygon.union(target).area),distance=float(np.linalg.norm(p[:2]-goal[:2])),angle=float(abs(np.arctan2(np.sin(a-goal[2]),np.cos(a-goal[2])))),z=float(p[2]),tilt=float(np.arccos(np.clip(mat[2,2],-1,1))))
    def reset_task(self,seed,difficulty='translation'):
        # Same positional/angular curriculum and deterministic random stream;
        # physical object dimensions and controller are a new environment.
        self.rng=np.random.default_rng(seed);self.placement_initializer.rng=self.rng;self.goal=np.zeros(3);self.reset()
        self.fixed_orientation=self.sim.data.site_xmat[self.robots[0].eef_site_id['right']].reshape(3,3).copy()
        rng=np.random.default_rng(seed);self.goal=np.r_[rng.uniform(-.035,.035,2),0.]
        gid=self.sim.model.body_name2id('goal');self.sim.model.body_pos[gid,:2]=self.goal[:2];self.sim.model.body_quat[gid]=[1,0,0,0]
        start=self.goal[:2]+[-.17,-.02]
        for t in range(300):
            self.move(np.clip((start-self.xy())/STEP,-1,1))
            if t>100 and np.linalg.norm(start-self.xy())<.0005:break
        direction=np.array([1.,rng.uniform(-.12,.12)]);direction/=np.linalg.norm(direction)
        distance=rng.uniform(.045,.09);angle=rng.uniform(-.12,.12);p=self.goal[:2]-direction*distance
        self.sim.data.set_joint_qpos(self.cube.joints[0],np.r_[p,.817,np.cos(angle/2),0,0,np.sin(angle/2)])
        self.sim.data.qvel[:]=0;self.sim.forward()
        for _ in range(8):self.move(np.zeros(2))
        self.timestep=0;self.done=False;return self.state_vector()

def make_env(images=False):
    cfg=load_composite_controller_config(controller='BASIC')
    cfg['body_parts']={'right':{'type':'OSC_POSE','input_max':1,'input_min':-1,'output_max':[.02,.02,.02,.5,.5,.5],'output_min':[-.02,-.02,-.02,-.5,-.5,-.5],'kp':300,'damping_ratio':1.2,'impedance_mode':'fixed','kp_limits':[0,300],'damping_ratio_limits':[0,10],'position_limits':None,'orientation_limits':None,'uncouple_pos_ori':True,'input_type':'delta','input_ref_frame':'world','interpolation':None,'ramp_ratio':.2,'gripper':{'type':'GRIP'}}}
    return PrecisionPushT(robots='Panda',gripper_types='CylinderPusher',controller_configs=cfg,initialization_noise=None,has_renderer=False,has_offscreen_renderer=images,use_camera_obs=False,use_object_obs=False,camera_names='birdview',camera_heights=128,camera_widths=128,control_freq=20,horizon=2000,ignore_done=True,hard_reset=False,seed=0,table_friction=(.12,.005,.0001))
