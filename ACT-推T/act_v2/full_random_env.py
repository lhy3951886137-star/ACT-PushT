"""Full-random initial position & full 360-degree [-pi, pi] initial angle Push-T environment.

Workspace is centered at Panda's true measured reachable table center [-0.07, 0.0]
(measured Panda OSC_POSE reach at z=0.825: X in [-0.28, +0.11], Y in [-0.27, +0.29]),
ensuring the pusher can physically orbit ALL 4 sides of the 15cm T-block for full 360-deg
rotations and arbitrary 2D translations.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon

from precision_env import PrecisionPushT, STEP, VERTICES, rotation
from robosuite.controllers import load_composite_controller_config

CAMERAS = ("birdview", "robot0_eye_in_hand")
# Centered directly over the reachable workspace [-0.07, 0.0] with fovy=38 deg (~0.58m FOV at table)
FULL_CAMERA_CONFIG = dict(
    birdview=dict(pos=[-0.07, 0.0, 1.65], fovy=38.0),
    robot0_eye_in_hand=dict(fovy=95.0),
)

WORKSPACE_CENTER = np.array([-0.07, 0.0], dtype=np.float64)
GOAL_X_RANGE = (-0.10, -0.04)     # 6cm random X around workspace center
GOAL_Y_RANGE = (-0.035, 0.035)    # 7cm random Y around workspace center
OBJ_X_RANGE = (-0.16, 0.01)       # 17cm completely random X (all 4 sides reachable by Panda)
OBJ_Y_RANGE = (-0.12, 0.12)       # 24cm completely random Y (all 360-deg directions around goal)
MIN_START_DIST = 0.03             # avoid trivial already-solved spawns


class FullRandomPushT(PrecisionPushT):
    def reset_task(self, seed: int, difficulty: str = "full_random"):
        self.rng = np.random.default_rng(seed)
        self.placement_initializer.rng = self.rng
        self.goal = np.zeros(3)
        self.reset()
        self.fixed_orientation = self.sim.data.site_xmat[self.robots[0].eef_site_id["right"]].reshape(3, 3).copy()

        rng = np.random.default_rng(seed)
        self.goal = np.array(
            [rng.uniform(*GOAL_X_RANGE), rng.uniform(*GOAL_Y_RANGE), 0.0],
            dtype=np.float64,
        )
        gid = self.sim.model.body_name2id("goal")
        self.sim.model.body_pos[gid, :2] = self.goal[:2]
        self.sim.model.body_quat[gid] = [1, 0, 0, 0]

        # Completely random initial T-block XY and full [-pi, pi] orientation
        for _ in range(100):
            p = np.array([rng.uniform(*OBJ_X_RANGE), rng.uniform(*OBJ_Y_RANGE)], dtype=np.float64)
            angle = float(rng.uniform(-np.pi, np.pi))
            if np.linalg.norm(p - self.goal[:2]) >= MIN_START_DIST or abs(angle) >= 0.35:
                break

        # Safe non-colliding pusher staging strictly within Panda's measured reachable zone
        R = rotation(angle)
        clearance = Polygon(VERTICES @ R.T + p).buffer(0.032, join_style=2)
        candidates = [
            np.array([-0.22, 0.0]),
            np.array([-0.07, -0.20]),
            np.array([-0.07, 0.20]),
            np.array([0.08, 0.0]),
            np.array([-0.20, -0.17]),
            np.array([-0.20, 0.17]),
            np.array([0.06, -0.17]),
            np.array([0.06, 0.17]),
        ]
        safe = [c for c in candidates if not clearance.contains(Point(c))]
        start = max(safe, key=lambda c: float(np.linalg.norm(c - p)))

        # Move T-block clear of the pusher warmup path so staging never collides or triggers elbow singularity
        self.sim.data.set_joint_qpos(
            self.cube.joints[0],
            np.array([0.35, 0.35, 0.817, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        self.sim.data.qvel[:] = 0
        self.sim.forward()

        for t in range(350):
            self.move(np.clip((start - self.xy()) / STEP, -1, 1))
            if t > 80 and np.linalg.norm(start - self.xy()) < 0.0008:
                break

        self.sim.data.set_joint_qpos(
            self.cube.joints[0],
            np.r_[p, 0.817, np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)],
        )
        self.sim.data.qvel[:] = 0
        self.sim.forward()
        for _ in range(8):
            self.move(np.zeros(2))
        self.timestep = 0
        self.done = False
        return self.state_vector()


def make_full_env(images: bool = False) -> FullRandomPushT:
    cfg = load_composite_controller_config(controller="BASIC")
    cfg["body_parts"] = {
        "right": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.02, 0.02, 0.02, 0.5, 0.5, 0.5],
            "output_min": [-0.02, -0.02, -0.02, -0.5, -0.5, -0.5],
            "kp": 300,
            "damping_ratio": 1.2,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "input_type": "delta",
            "input_ref_frame": "world",
            "interpolation": None,
            "ramp_ratio": 0.2,
            "gripper": {"type": "GRIP"},
        }
    }
    env = FullRandomPushT(
        robots="Panda",
        gripper_types="CylinderPusher",
        controller_configs=cfg,
        initialization_noise=None,
        has_renderer=False,
        has_offscreen_renderer=images,
        use_camera_obs=False,
        use_object_obs=False,
        camera_names="birdview",
        camera_heights=128,
        camera_widths=128,
        control_freq=20,
        horizon=2400,
        ignore_done=True,
        hard_reset=False,
        seed=0,
        table_friction=(0.12, 0.005, 0.0001),
    )
    if images:
        for name, ccfg in FULL_CAMERA_CONFIG.items():
            cid = env.sim.model.camera_name2id(name)
            if "pos" in ccfg:
                env.sim.model.cam_pos[cid] = ccfg["pos"]
            env.sim.model.cam_fovy[cid] = ccfg["fovy"]
        env.sim.forward()
    return env


def observe_full(env: FullRandomPushT) -> dict:
    """Pure visual observation: robot_xy[2] + dual 128x128 RGB cameras."""
    images = np.stack([env.sim.render(width=128, height=128, camera_name=c)[::-1].copy() for c in CAMERAS])
    return dict(robot_xy=np.asarray(env.xy(), np.float32), images=images)
