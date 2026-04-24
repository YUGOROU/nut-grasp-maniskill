"""
NutGrasp-v1: SO-101 picks a nut (almond or ginko) from a tray.

Scene layout (world frame, Z-up):
  - Table surface at z = 0
  - Tray (255x202x26mm) centered at table center
  - Two bowls (120mm dia) for sorted output
  - One nut spawned randomly inside tray
"""

from __future__ import annotations
import os
from typing import Any, Dict

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.building.ground import build_ground

# Robot agent must be imported to trigger @register_agent
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "assets", "robot"))
import so101_agent  # noqa: F401

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


def _load_obj_as_static(scene, obj_path: str, name: str, pose: sapien.Pose):
    builder = scene.create_actor_builder()
    builder.set_initial_pose(pose)
    builder.add_visual_from_file(obj_path)
    builder.add_nonconvex_collision_from_file(obj_path)
    actor = builder.build_static(name=name)
    return actor


def _load_glb_dynamic(scene, glb_path: str, name: str, pose: sapien.Pose, scale=1.0):
    builder = scene.create_actor_builder()
    builder.set_initial_pose(pose)
    builder.add_visual_from_file(glb_path, scale=[scale] * 3)
    builder.add_multiple_convex_collisions_from_file(glb_path, scale=[scale] * 3)
    actor = builder.build(name=name)
    return actor


@register_env("NutGrasp-v1", max_episode_steps=200)
class NutGraspEnv(BaseEnv):
    """SO-101 nut grasping task for sim-to-real transfer."""

    SUPPORTED_ROBOTS = ["so101"]

    # Table dimensions (m)
    TABLE_SIZE   = (0.85, 0.54, 0.02)
    # Tray center on table (relative to table center)
    TRAY_OFFSET  = (0.0, 0.0, 0.013)     # z = tray half-height
    TRAY_HALF    = (0.1275, 0.101, 0.013)
    # Bowl positions (almond left, ginko right)
    BOWL_OFFSET  = [(-0.18, 0.0, 0.03), (0.18, 0.0, 0.03)]
    # Robot base relative to table center
    ROBOT_OFFSET = (-0.35, 0.0, 0.02)

    @property
    def _default_sensor_configs(self):
        # Overhead camera looking down at the tray
        pose = sapien.Pose(p=[0.0, 0.0, 0.50],
                           q=[0.707, -0.707, 0.0, 0.0])
        return [CameraConfig("overhead_cam", pose=pose,
                             width=64, height=64,
                             fov=np.deg2rad(60), near=0.01, far=10.0)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien.Pose(p=[-0.4, -0.4, 0.5],
                           q=[0.924, -0.383, 0.0, 0.0])
        return [CameraConfig("render_cam", pose=pose,
                             width=512, height=512,
                             fov=np.deg2rad(60), near=0.01, far=10.0)]

    def __init__(self, *args, robot_uids="so101", nut_type="almond", **kwargs):
        assert nut_type in ("almond", "ginko"), "nut_type must be 'almond' or 'ginko'"
        self.nut_type = nut_type
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        robot_pose = sapien.Pose(p=list(self.ROBOT_OFFSET))
        super()._load_agent(options, robot_pose)

    def _load_scene(self, options: dict):
        # --- Ground plane ---
        build_ground(self.scene, altitude=0.0)

        # --- Table ---
        table_path = os.path.join(ASSET_DIR, "objects", "table", "table.obj")
        # Table sits on ground: surface at z=0, centre at z = -half_thickness
        table_z = -self.TABLE_SIZE[2] / 2
        self.table = _load_obj_as_static(
            self.scene, table_path, "table",
            sapien.Pose(p=[0, 0, table_z])
        )

        # --- Tray ---
        tray_path = os.path.join(ASSET_DIR, "objects", "tray", "tray.obj")
        self.tray = _load_obj_as_static(
            self.scene, tray_path, "tray",
            sapien.Pose(p=list(self.TRAY_OFFSET))
        )

        # --- Bowls (static for now) ---
        self.bowls = []
        for i, (bname, offset) in enumerate(
            zip(("almond_bowl", "ginko_bowl"), self.BOWL_OFFSET)
        ):
            bowl_path = os.path.join(ASSET_DIR, "objects", "bowls", f"{bname}.obj")
            b = _load_obj_as_static(
                self.scene, bowl_path, bname,
                sapien.Pose(p=list(offset))
            )
            self.bowls.append(b)

        # --- Nut (dynamic) ---
        if self.nut_type == "almond":
            glb = os.path.join(ASSET_DIR, "objects", "almond", "almond_single.glb")
        else:
            glb = os.path.join(ASSET_DIR, "objects", "ginko", "Ginko_single.glb")

        nut_init_z = self.TRAY_OFFSET[2] + self.TRAY_HALF[2] + 0.015
        self.nut = _load_glb_dynamic(
            self.scene, glb, f"{self.nut_type}_nut",
            sapien.Pose(p=[0, 0, nut_init_z])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # Randomize nut position inside tray (80% of tray area)
            xy = (torch.rand((b, 2)) * 2 - 1) * torch.tensor(
                [self.TRAY_HALF[0] * 0.8, self.TRAY_HALF[1] * 0.8]
            )
            z = torch.full((b, 1), self.TRAY_OFFSET[2] + self.TRAY_HALF[2] + 0.015)
            pos = torch.cat([xy, z], dim=-1)
            nut_pose = Pose.create_from_pq(p=pos, q=torch.tensor([1., 0., 0., 0.]).expand(b, -1))
            self.nut.set_pose(nut_pose)

    def evaluate(self) -> Dict[str, Any]:
        nut_pos   = self.nut.pose.p          # (B, 3)
        # Target bowl index: almond=0, ginko=1
        target_idx = 0 if self.nut_type == "almond" else 1
        bowl_pos = torch.tensor(self.BOWL_OFFSET[target_idx], device=self.device)
        dist = torch.linalg.norm(nut_pos - bowl_pos, dim=-1)
        # Success: nut within 5cm of target bowl center and above z=0
        success = (dist < 0.05) & (nut_pos[:, 2] > 0.005)
        return {"success": success, "dist_to_bowl": dist}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_pos  = self.agent.tcp.pose.p     # (B, 3)
        nut_pos  = self.nut.pose.p           # (B, 3)
        target_idx = 0 if self.nut_type == "almond" else 1
        bowl_pos = torch.tensor(self.BOWL_OFFSET[target_idx], device=self.device)

        # Stage 1: reach nut
        reach_dist  = torch.linalg.norm(tcp_pos - nut_pos, dim=-1)
        reach_rew   = 1 - torch.tanh(5.0 * reach_dist)

        # Stage 2: grasp — reward closing gripper while near nut
        # gripper joint: 0=closed, 0.04m=open (normalize_action=True maps [-1,1]->[0,0.04])
        gripper_q  = self.agent.robot.get_qpos()[:, -1].clamp(0.0, 0.04)
        gripper_open = gripper_q / 0.04          # [0=closed, 1=open]
        grasp_rew  = (1.0 - gripper_open) * reach_rew

        # Stage 3: lift nut (z above tray rim)
        lift_height = (nut_pos[:, 2] - (self.TRAY_OFFSET[2] + self.TRAY_HALF[2])).clamp(0)
        lift_rew    = torch.tanh(10.0 * lift_height)

        # Stage 4: place nut in target bowl
        place_dist  = torch.linalg.norm(nut_pos - bowl_pos, dim=-1)
        place_rew   = 1 - torch.tanh(5.0 * place_dist)

        # max = 1 + 2 + 2 + 3 = 8
        reward = reach_rew + grasp_rew * 2.0 + lift_rew * 2.0 + place_rew * 3.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 8.0
