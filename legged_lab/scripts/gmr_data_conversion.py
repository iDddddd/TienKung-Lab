import pickle
import numpy as np
import argparse
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Pure-numpy quaternion helpers (wxyz convention throughout)
# No Isaac Lab / Isaac Sim dependency so this script runs without a GPU.
# ---------------------------------------------------------------------------


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """q: (..., 4) wxyz  →  conjugate wxyz"""
    out = q.copy()
    out[..., 1:] *= -1
    return out


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for (N,4) wxyz arrays."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def _axis_angle_from_quat(q: np.ndarray) -> np.ndarray:
    """Convert (N,4) wxyz unit quaternion to (N,3) axis-angle vector."""
    vec = q[:, 1:]                                          # (N, 3)
    mag = np.linalg.norm(vec, axis=-1, keepdims=True)      # (N, 1)
    angle = 2.0 * np.arctan2(mag[:, 0], q[:, 0])           # (N,)
    safe_mag = np.where(mag[:, 0] > 1e-8, mag[:, 0], 1.0)
    axis = vec / safe_mag[:, None]
    return axis * angle[:, None]


# ---------------------------------------------------------------------------
# Robot configurations
#
# DISPLAY format per frame (AMPLoaderDisplay input):
#   root_pos(3) + root_euler_XYZ(3) + dof_pos(N) + lin_vel(3) + ang_vel(3) + dof_vel(N)
#
# For both tienkung and F1 the URDF joint order already matches the display
# format joint order, so no index-remapping is required.
#   tienkung: left_leg(6) right_leg(6) left_arm(4) right_arm(4)   → N=20
#   F1:       left_leg(6) right_leg(6) waist(3) left_arm(8) right_arm(8) → N=31
# ---------------------------------------------------------------------------

ROBOT_CONFIGS = {
    "tienkung": {
        "num_dof": 20,
        "display_idx": list(range(20)),   # identity — URDF order = display order
        "task_name": "walk",
        "description": "TienKung2 Lite (20 DOF): 6L-leg + 6R-leg + 4L-arm + 4R-arm",
    },
    "f1": {
        "num_dof": 31,
        # URDF order:
        #  0-5  : left_hip_roll/pitch/yaw, left_knee, left_ankle_pitch/roll
        #  6-11 : right_hip_roll/pitch/yaw, right_knee, right_ankle_pitch/roll
        #  12-14: waist_pitch/roll/yaw
        #  15-22: left_scapula_roll, left_shoulder_pitch/roll/yaw,
        #         left_elbow, left_wrist_yaw/roll/pitch
        #  23-30: right_scapula_roll, right_shoulder_pitch/roll/yaw,
        #         right_elbow, right_wrist_yaw/roll/pitch
        "display_idx": list(range(31)),   # identity — URDF order = display order
        "task_name": "f1_walk",
        "description": "F1 humanoid (31 DOF): 6L-leg + 6R-leg + 3-waist + 8L-arm + 8R-arm",
    },
}


def _write_txt_format(data: np.ndarray, output_txt: str, fps: float) -> None:
    """Write (F, cols) numpy array to the JSON-like AMP motion TXT format."""
    np.savetxt(output_txt, data, fmt="%f", delimiter=", ")
    with open(output_txt, "r") as f:
        lines = f.readlines()
    n = len(lines)
    with open(output_txt, "w") as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0 / fps:.4f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')
        for i, line in enumerate(lines):
            suffix = "],\n" if i < n - 1 else "]\n"
            f.write("  [" + line.rstrip() + suffix)
        f.write("]\n}")


def convert_pkl_to_custom(input_pkl: str, output_txt: str, fps: float, robot: str = "tienkung") -> None:
    if robot not in ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot '{robot}'. Choose from: {list(ROBOT_CONFIGS.keys())}")

    cfg = ROBOT_CONFIGS[robot]

    with open(input_pkl, "rb") as f:
        motion_data = pickle.load(f)

    # Auto-use PKL fps when available
    if "fps" in motion_data and motion_data["fps"] is not None:
        pkl_fps = float(motion_data["fps"])
        if abs(pkl_fps - fps) > 0.5:
            print(f"[INFO] PKL contains fps={pkl_fps:.0f}; overriding --fps {fps:.0f} → {pkl_fps:.0f}")
            fps = pkl_fps
    dt = 1.0 / fps

    root_pos      = np.array(motion_data["root_pos"], dtype=np.float64)   # (T, 3)
    root_rot_xyzw = np.array(motion_data["root_rot"], dtype=np.float64)   # (T, 4)  xyzw
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]                        # → wxyz
    dof_pos_raw   = np.array(motion_data["dof_pos"],  dtype=np.float64)   # (T, N)

    # Validate DOF count
    if dof_pos_raw.shape[1] != cfg["num_dof"]:
        raise ValueError(
            f"PKL dof_pos has {dof_pos_raw.shape[1]} joints, "
            f"but '{robot}' expects {cfg['num_dof']}."
        )

    # Reorder joints for display format (identity mapping for both robots)
    dof_pos = dof_pos_raw[:, cfg["display_idx"]]                          # (T, N)

    # ---------- Finite-difference derivatives ----------
    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt                   # (T-1, 3)

    q_conj  = _quat_conjugate(root_rot_wxyz[:-1])
    dq      = _quat_mul(q_conj, root_rot_wxyz[1:])
    root_ang_vel = _axis_angle_from_quat(dq) / dt                        # (T-1, 3)

    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt                          # (T-1, N)

    # Euler angles from xyzw quaternion, unwrapped to avoid 2π jumps
    euler = Rotation.from_quat(root_rot_xyzw[:-1]).as_euler("XYZ", degrees=False)
    euler = np.unwrap(euler, axis=0)                                      # (T-1, 3)

    # ---------- Assemble display format ----------
    # root_pos(3) | euler_XYZ(3) | dof_pos(N) | lin_vel(3) | ang_vel(3) | dof_vel(N)
    data_out = np.concatenate(
        (root_pos[:-1], euler, dof_pos[:-1], root_lin_vel, root_ang_vel, dof_vel),
        axis=1,
    )
    n_frames, n_cols = data_out.shape
    expected = 6 + 2 * cfg["num_dof"] + 6
    assert n_cols == expected, f"Expected {expected} cols, got {n_cols}"

    _write_txt_format(data_out, output_txt, fps)

    print(f"Robot   : {robot} — {cfg['description']}")
    print(f"Input   : {input_pkl}  (T={n_frames + 1}, fps={fps:.0f})")
    print(f"Output  : {output_txt}  ({n_frames} frames × {n_cols} cols)")
    print(f"✅ Visualization TXT written successfully.")
    print()
    print("── Step 2: generate training TXT (FK end-effectors, requires simulation) ──")
    print(f"   Place the file above at the path in amp_motion_files_display, then run:")
    print(f"   python legged_lab/scripts/play_amp_animation.py \\")
    print(f"       --task {cfg['task_name']} --headless \\")
    print(f"       --save_path <path/to/motion_amp_expert/walk_eight.txt>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert a GMR PKL motion file to AMP *visualization* TXT format.\n"
            "The output is used by AMPLoaderDisplay (motion_visualization/).\n"
            "Run play_amp_animation.py afterwards to generate the training TXT\n"
            "(motion_amp_expert/) with FK-computed end-effector positions."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input_pkl",  type=str, required=True,
                        help="Input .pkl file")
    parser.add_argument("--output_txt", type=str, required=True,
                        help="Output visualization .txt file")
    parser.add_argument("--fps",   type=float, default=50.0,
                        help="Output FPS (default: 30; auto-detected from PKL when present)")
    parser.add_argument("--robot", type=str, default="tienkung",
                        choices=list(ROBOT_CONFIGS.keys()),
                        help="Target robot (default: tienkung)")
    args = parser.parse_args()

    convert_pkl_to_custom(args.input_pkl, args.output_txt, args.fps, args.robot)


# ---------------------------------------------------------------------------
# Robot configurations
#
# For the DISPLAY format (AMPLoaderDisplay) each frame layout is:
#   root_pos(3) + root_euler(3) + dof_pos(N) + lin_vel(3) + ang_vel(3) + dof_vel(N)
# where dof_pos columns must follow the "display order" below.
#
# For tienkung: display order = URDF order  (20 joints, identity mapping)
# For F1:       display order = URDF order  (31 joints, identity mapping)
#   URDF:  left_leg(0:6) | right_leg(6:12) | waist(12:15) | left_arm(15:23) | right_arm(23:31)
#   Display: same ordering → no reindex needed
# ---------------------------------------------------------------------------

ROBOT_CONFIGS = {
    "tienkung": {
        "num_dof": 20,
        # Identity: URDF order already matches display order for tienkung
        "display_idx": list(range(20)),
        "task_name": "walk",
        "description": "TienKung2 Lite (20 DOF): 6L-leg + 6R-leg + 4L-arm + 4R-arm",
    },
    "f1": {
        "num_dof": 31,
        # Identity: URDF order already matches display order for F1
        #   0-5:  left_hip_roll/pitch/yaw, left_knee, left_ankle_pitch/roll
        #   6-11: right_hip_roll/pitch/yaw, right_knee, right_ankle_pitch/roll
        #   12-14: waist_pitch/roll/yaw
        #   15-22: left_scapula_roll, left_shoulder_pitch/roll/yaw, left_elbow,
        #          left_wrist_yaw/roll/pitch
        #   23-30: right_scapula_roll, right_shoulder_pitch/roll/yaw, right_elbow,
        #          right_wrist_yaw/roll/pitch
        "display_idx": list(range(31)),
        "task_name": "f1_walk",
        "description": "F1 humanoid (31 DOF): 6L-leg + 6R-leg + 3-waist + 8L-arm + 8R-arm",
    },
}


def _write_txt_format(data: np.ndarray, output_txt: str, fps: float) -> None:
    """Write (N, cols) numpy array to the JSON-like AMP motion TXT format."""
    np.savetxt(output_txt, data, fmt="%f", delimiter=", ")
    with open(output_txt, "r") as f:
        lines = f.readlines()
    n = len(lines)
    with open(output_txt, "w") as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0 / fps:.4f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')
        for i, line in enumerate(lines):
            suffix = "],\n" if i < n - 1 else "]\n"
            f.write("  [" + line.rstrip() + suffix)
        f.write("]\n}")




