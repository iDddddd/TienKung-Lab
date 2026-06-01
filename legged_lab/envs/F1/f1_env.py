# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""F1 humanoid environment (31-DOF, AMP training).

AMP observation layout (74-dim):
  Training   (AMPLoader / F1AMPLoader):
    [0:8]    right_arm_dof_pos  (8)
    [8:16]   left_arm_dof_pos   (8)
    [16:22]  right_leg_dof_pos  (6)
    [22:28]  left_leg_dof_pos   (6)
    [28:31]  waist_dof_pos      (3)
    [31:39]  right_arm_dof_vel  (8)
    [39:47]  left_arm_dof_vel   (8)
    [47:53]  right_leg_dof_vel  (6)
    [53:59]  left_leg_dof_vel   (6)
    [59:62]  waist_dof_vel      (3)
    [62:65]  left_hand_pos      (3)
    [65:68]  right_hand_pos     (3)
    [68:71]  left_foot_pos      (3)
    [71:74]  right_foot_pos     (3)

  Visualization (AMPLoaderDisplay / F1AMPLoaderDisplay):
    [0:3]    root_pos           (3)
    [3:6]    root_euler_XYZ     (3)
    [6:12]   left_leg_pos       (6)
    [12:18]  right_leg_pos      (6)
    [18:21]  waist_pos          (3)
    [21:29]  left_arm_pos       (8)
    [29:37]  right_arm_pos      (8)
    [37:40]  lin_vel            (3)
    [40:43]  ang_vel            (3)
    [43:49]  left_leg_vel       (6)
    [49:55]  right_leg_vel      (6)
    [55:58]  waist_vel          (3)
    [58:66]  left_arm_vel       (8)
    [66:74]  right_arm_vel      (8)
"""

import torch
from scipy.spatial.transform import Rotation

from isaaclab.utils.math import quat_apply, quat_conjugate

from legged_lab.envs.F1.walk_cfg import F1WalkFlatEnvCfg
from legged_lab.envs.tienkung.tienkung_env import TienKungEnv
from rsl_rl.utils import AMPLoader, AMPLoaderDisplay


# ---------------------------------------------------------------------------
# AMP Loader sub-classes — override class-level size constants for 31-DOF F1
# ---------------------------------------------------------------------------


class F1AMPLoader(AMPLoader):
    """Motion loader for F1 training data (74-column files).

    Layout per frame: 31 joint_pos | 31 joint_vel | 12 end-effector pos
    """

    JOINT_POS_SIZE = 31
    JOINT_VEL_SIZE = 31
    END_EFFECTOR_POS_SIZE = 12

    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE       # 31
    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX                          # 31
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE         # 62
    END_POS_START_IDX = JOINT_VEL_END_IDX                             # 62
    END_POS_END_IDX = END_POS_START_IDX + END_EFFECTOR_POS_SIZE       # 74


class F1AMPLoaderDisplay(AMPLoaderDisplay):
    """Motion loader for F1 visualization data (74-column files).

    Layout per frame:
      pose part  [0:37]  = root_pos(3) + root_euler(3) + joints(31)
      vel  part  [37:74] = lin_vel(3)  + ang_vel(3)    + joint_vels(31)
    """

    JOINT_POS_SIZE = 37   # root_pos(3) + root_euler(3) + 31 joints
    JOINT_VEL_SIZE = 37   # lin_vel(3)  + ang_vel(3)    + 31 joint vels

    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE       # 37
    ROOT_STATES_NUM = 6
    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX                          # 37
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE         # 74


# ---------------------------------------------------------------------------
# F1 Environment
# ---------------------------------------------------------------------------


class F1Env(TienKungEnv):
    """Isaac Lab environment for the F1 humanoid robot.

    Subclasses TienKungEnv and overrides joint ID lookups, visualization,
    and AMP observation construction to match F1's 31-DOF layout.
    """

    def __init__(self, cfg: F1WalkFlatEnvCfg, headless: bool):
        # Parent __init__ builds the scene, calls init_buffers(), and creates
        # a TienKung AMPLoaderDisplay.  We replace it afterwards.
        super().__init__(cfg, headless)

        # Replace the tienkung-specific display loader with the F1 version.
        self.amp_loader_display = F1AMPLoaderDisplay(
            motion_files=self.cfg.amp_motion_files_display,
            device=self.device,
            time_between_frames=self.physics_dt,
        )
        self.motion_len = self.amp_loader_display.trajectory_num_frames[0]

    # ------------------------------------------------------------------
    # Joint/body ID initialisation
    # ------------------------------------------------------------------

    def _init_body_joint_ids(self):
        """F1-specific body / joint ID lookups, called from TienKungEnv.init_buffers()."""
        # Foot contact bodies
        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        # Hand-proxy bodies (last wrist link)
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_wrist_pitch_link", "right_wrist_pitch_link"],
            preserve_order=True,
        )

        # Leg joints: roll, pitch, yaw, knee, ankle_pitch, ankle_roll
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_hip_roll_joint",
                "left_hip_pitch_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_hip_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ],
            preserve_order=True,
        )

        # Waist joints (F1 specific)
        self.waist_ids, _ = self.robot.find_joints(
            name_keys=["waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint"],
            preserve_order=True,
        )

        # Arm joints: scapula, shoulder x3, elbow, wrist x3
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_scapula_roll_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_yaw_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
            ],
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_scapula_roll_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_yaw_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
            ],
            preserve_order=True,
        )

        # Ankle joints
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_ankle_pitch_joint",
                "right_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_roll_joint",
            ],
            preserve_order=True,
        )

    def init_buffers(self):
        """Call parent init (which now invokes our _init_body_joint_ids), then apply F1 overrides."""
        super().init_buffers()
        # F1 wrist links are at the hand; no forearm offset needed.
        self.left_arm_local_vec = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_arm_local_vec = torch.zeros((self.num_envs, 3), device=self.device)

    # ------------------------------------------------------------------
    # Motion visualisation
    # ------------------------------------------------------------------

    def visualize_motion(self, time: float):
        """Play back an F1 display motion frame at the given time.

        F1 display frame layout (74 cols):
          [0:3]   root_pos       [3:6]   root_euler
          [6:12]  left_leg_pos   [12:18] right_leg_pos
          [18:21] waist_pos      [21:29] left_arm_pos  [29:37] right_arm_pos
          [37:40] lin_vel        [40:43] ang_vel
          [43:49] left_leg_vel   [49:55] right_leg_vel
          [55:58] waist_vel      [58:66] left_arm_vel  [66:74] right_arm_vel
        """
        frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        device = self.device

        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)

        # Positions
        dof_pos[:, self.left_leg_ids]  = frame[6:12]
        dof_pos[:, self.right_leg_ids] = frame[12:18]
        dof_pos[:, self.waist_ids]     = frame[18:21]
        dof_pos[:, self.left_arm_ids]  = frame[21:29]
        dof_pos[:, self.right_arm_ids] = frame[29:37]

        # Velocities
        dof_vel[:, self.left_leg_ids]  = frame[43:49]
        dof_vel[:, self.right_leg_ids] = frame[49:55]
        dof_vel[:, self.waist_ids]     = frame[55:58]
        dof_vel[:, self.left_arm_ids]  = frame[58:66]
        dof_vel[:, self.right_arm_ids] = frame[66:74]

        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)

        env_ids = torch.arange(self.num_envs, device=device)

        root_pos = frame[:3].clone()
        root_pos[2] += 0.3

        euler = frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=torch.float32,
            device=device,
        )

        lin_vel = frame[37:40].clone()
        ang_vel = torch.zeros_like(lin_vel)

        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3]  = root_pos.unsqueeze(0).expand(self.num_envs, -1)
        root_state[:, 3:7]  = quat_wxyz.unsqueeze(0).expand(self.num_envs, -1)
        root_state[:, 7:10] = lin_vel.unsqueeze(0).expand(self.num_envs, -1)
        root_state[:, 10:13] = ang_vel.unsqueeze(0).expand(self.num_envs, -1)

        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)

        # Compute end-effector positions in root frame
        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(
                self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7],
                self.left_arm_local_vec,
            )
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(
                self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7],
                self.right_arm_local_vec,
            )
        )
        left_hand_pos  = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)

        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos  = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)

        # Cache for get_amp_obs_for_expert_trans
        self.left_leg_dof_pos  = dof_pos[:, self.left_leg_ids]
        self.right_leg_dof_pos = dof_pos[:, self.right_leg_ids]
        self.left_leg_dof_vel  = dof_vel[:, self.left_leg_ids]
        self.right_leg_dof_vel = dof_vel[:, self.right_leg_ids]
        self.left_arm_dof_pos  = dof_pos[:, self.left_arm_ids]
        self.right_arm_dof_pos = dof_pos[:, self.right_arm_ids]
        self.left_arm_dof_vel  = dof_vel[:, self.left_arm_ids]
        self.right_arm_dof_vel = dof_vel[:, self.right_arm_ids]
        self.waist_dof_pos     = dof_pos[:, self.waist_ids]
        self.waist_dof_vel     = dof_vel[:, self.waist_ids]

        return torch.cat(
            (
                self.right_arm_dof_pos,   # 8
                self.left_arm_dof_pos,    # 8
                self.right_leg_dof_pos,   # 6
                self.left_leg_dof_pos,    # 6
                self.waist_dof_pos,       # 3
                self.right_arm_dof_vel,   # 8
                self.left_arm_dof_vel,    # 8
                self.right_leg_dof_vel,   # 6
                self.left_leg_dof_vel,    # 6
                self.waist_dof_vel,       # 3
                left_hand_pos,            # 3
                right_hand_pos,           # 3
                left_foot_pos,            # 3
                right_foot_pos,           # 3
            ),
            dim=-1,
        )  # total = 74

    # ------------------------------------------------------------------
    # AMP observation for the discriminator
    # ------------------------------------------------------------------

    def get_amp_obs_for_expert_trans(self) -> torch.Tensor:
        """Return the 74-dim AMP observation from the current sim state.

        Layout (matches training motion-file columns):
          [0:8]   right_arm_pos  [8:16]  left_arm_pos
          [16:22] right_leg_pos  [22:28] left_leg_pos  [28:31] waist_pos
          [31:39] right_arm_vel  [39:47] left_arm_vel
          [47:53] right_leg_vel  [53:59] left_leg_vel  [59:62] waist_vel
          [62:65] left_hand_pos  [65:68] right_hand_pos
          [68:71] left_foot_pos  [71:74] right_foot_pos
        """
        # End-effector positions in root frame
        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(
                self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7],
                self.left_arm_local_vec,
            )
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(
                self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7],
                self.right_arm_local_vec,
            )
        )
        left_hand_pos  = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)

        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos  = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)

        # Joint states
        self.left_leg_dof_pos  = self.robot.data.joint_pos[:, self.left_leg_ids]
        self.right_leg_dof_pos = self.robot.data.joint_pos[:, self.right_leg_ids]
        self.left_leg_dof_vel  = self.robot.data.joint_vel[:, self.left_leg_ids]
        self.right_leg_dof_vel = self.robot.data.joint_vel[:, self.right_leg_ids]
        self.left_arm_dof_pos  = self.robot.data.joint_pos[:, self.left_arm_ids]
        self.right_arm_dof_pos = self.robot.data.joint_pos[:, self.right_arm_ids]
        self.left_arm_dof_vel  = self.robot.data.joint_vel[:, self.left_arm_ids]
        self.right_arm_dof_vel = self.robot.data.joint_vel[:, self.right_arm_ids]
        self.waist_dof_pos     = self.robot.data.joint_pos[:, self.waist_ids]
        self.waist_dof_vel     = self.robot.data.joint_vel[:, self.waist_ids]

        return torch.cat(
            (
                self.right_arm_dof_pos,   # 8
                self.left_arm_dof_pos,    # 8
                self.right_leg_dof_pos,   # 6
                self.left_leg_dof_pos,    # 6
                self.waist_dof_pos,       # 3
                self.right_arm_dof_vel,   # 8
                self.left_arm_dof_vel,    # 8
                self.right_leg_dof_vel,   # 6
                self.left_leg_dof_vel,    # 6
                self.waist_dof_vel,       # 3
                left_hand_pos,            # 3
                right_hand_pos,           # 3
                left_foot_pos,            # 3
                right_foot_pos,           # 3
            ),
            dim=-1,
        )  # total = 74
