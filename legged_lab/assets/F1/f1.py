# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Configuration for the F1 humanoid robot.

The F1 robot has 31 active revolute joints:
  - 6 left leg  + 6 right leg  = 12 leg joints
  - 3 waist joints
  - 8 left arm  + 8 right arm  = 16 arm joints (scapula, shoulder x3, elbow, wrist x3)

URDF: F1_simple_collision.urdf
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

F1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_ASSET_DIR}/F1/urdf/F1_simple_collision.urdf",
        fix_base=False,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        joint_drive=None,  # ImplicitActuatorCfg handles PD gains; skip URDF-level drive setup
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # disabled for RL training stability (same as TienKung)
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            # Left leg — defaults aligned with walk_eight expert motion range
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.2,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.5,
            "left_ankle_pitch_joint": -0.3,   # expert mean ≈ -0.26; MUST be negative
            "left_ankle_roll_joint": 0.0,
            # Right leg
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.2,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.5,
            "right_ankle_pitch_joint": -0.3,  # expert mean ≈ -0.28; MUST be negative
            "right_ankle_roll_joint": 0.0,
            # Waist
            "waist_pitch_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            # Left arm — shoulder_roll aligned with expert (~-1.1 rad abduction)
            "left_scapula_roll_joint": 0.1,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": -1.0,  # expert mean ≈ -1.12
            "left_shoulder_yaw_joint": 0.4,    # expert mean ≈ 0.43
            "left_elbow_joint": -0.2,
            "left_wrist_yaw_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            # Right arm
            "right_scapula_roll_joint": -0.1,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 1.0,  # expert mean ≈ 1.07
            "right_shoulder_yaw_joint": 0.5,   # expert mean ≈ 0.53
            "right_elbow_joint": 0.1,
            "right_wrist_yaw_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_roll_joint": 40.0,
                ".*_hip_pitch_joint": 40.0,
                ".*_hip_yaw_joint": 27.0,
                ".*_knee_joint": 40.0,
            },
            velocity_limit_sim=50.0,
            stiffness={
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_hip_yaw_joint": 120.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_roll_joint": 8.0,
                ".*_hip_pitch_joint": 8.0,
                ".*_hip_yaw_joint": 6.0,
                ".*_knee_joint": 6.0,
            },
            armature=0.03,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 15.0,
                ".*_ankle_roll_joint": 20.0,
            },
            velocity_limit_sim=50.0,
            stiffness={
                ".*_ankle_pitch_joint": 30.0,
                ".*_ankle_roll_joint": 30.0,
            },
            damping={
                ".*_ankle_pitch_joint": 1.0,
                ".*_ankle_roll_joint": 3.0,
            },
            armature=0.03,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_pitch_joint",
                "waist_roll_joint",
                "waist_yaw_joint",
            ],
            effort_limit_sim={
                "waist_pitch_joint": 30.0,
                "waist_roll_joint": 40.0,
                "waist_yaw_joint": 27.0,
            },
            velocity_limit_sim=50.0,
            stiffness={
                "waist_pitch_joint": 40.0,
                "waist_roll_joint": 45.0,
                "waist_yaw_joint": 60.0,
            },
            damping={
                "waist_pitch_joint": 4.0,
                "waist_roll_joint": 4.0,
                "waist_yaw_joint": 4.0,
            },
            armature=0.03,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_scapula_roll_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
            ],
            effort_limit_sim={
                ".*_scapula_roll_joint": 20.0,
                ".*_shoulder_pitch_joint": 20.0,
                ".*_shoulder_roll_joint": 20.0,
                ".*_shoulder_yaw_joint": 16.0,
                ".*_elbow_joint": 16.0,
                ".*_wrist_yaw_joint": 16.0,
                ".*_wrist_roll_joint": 3.0,
                ".*_wrist_pitch_joint": 3.0,
            },
            velocity_limit_sim=50.0,
            stiffness={
                ".*_scapula_roll_joint": 25.0,
                ".*_shoulder_pitch_joint": 15.0,
                ".*_shoulder_roll_joint": 15.0,
                ".*_shoulder_yaw_joint": 15.0,
                ".*_elbow_joint": 15.0,
                ".*_wrist_yaw_joint": 15.0,
                ".*_wrist_roll_joint": 3.0,
                ".*_wrist_pitch_joint": 3.0,
            },
            damping={
                ".*_scapula_roll_joint": 2.0,
                ".*_shoulder_pitch_joint": 1.0,
                ".*_shoulder_roll_joint": 1.0,
                ".*_shoulder_yaw_joint": 1.0,
                ".*_elbow_joint": 1.0,
                ".*_wrist_yaw_joint": 0.5,
                ".*_wrist_roll_joint": 0.3,
                ".*_wrist_pitch_joint": 0.3,
            },
            armature=0.03,
        ),
    },
)
