"""SO-101 robot agent for ManiSkill3."""
import os
import numpy as np
import sapien
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDJointPosController,
    PDJointPosControllerConfig,
    PassiveControllerConfig,
)
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils


ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(ASSET_DIR, "so101.urdf")


@register_agent()
class SO101(BaseAgent):
    uid = "so101"
    urdf_path = URDF_PATH
    urdf_config = {
        "self_collisions": False,
    }

    # Joint names in URDF order
    arm_joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]
    gripper_joint_names = [
        "gripper",
    ]

    # Home pose: arm upright, gripper open
    keyframes = {
        "rest": Keyframe(
            qpos=np.deg2rad([0, -30, 60, -30, 0, 0]),
            pose=sapien.Pose(p=[0, 0, 0]),
        )
    }

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=100,
            damping=10,
            normalize_action=False,
        )
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=self.gripper_joint_names,
            lower=0.0,
            upper=0.04,
            stiffness=200,
            damping=20,
            normalize_action=True,
        )
        return {
            "pd_joint_pos": dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
            )
        }

    def _after_loading_articulation(self):
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "gripper_frame_link"
        )
