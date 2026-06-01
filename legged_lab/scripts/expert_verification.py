import argparse
import csv
import json
from pathlib import Path

import numpy as np


AMP_JOINT_NAMES_TIENKUNG = [
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "hip_roll_r_joint",
    "hip_pitch_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "hip_roll_l_joint",
    "hip_pitch_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
]

# F1 expert format (74 cols):
#   right_arm_pos(8) + left_arm_pos(8) + right_leg_pos(6) + left_leg_pos(6) + waist_pos(3)  = 31
#   right_arm_vel(8) + left_arm_vel(8) + right_leg_vel(6) + left_leg_vel(6) + waist_vel(3)  = 31
#   left_hand_pos(3) + right_hand_pos(3) + left_foot_pos(3) + right_foot_pos(3)             = 12
AMP_JOINT_NAMES_F1 = [
    "right_scapula_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "left_scapula_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
]

# Robot-specific column layout for expert (training) format
ROBOT_CONFIGS = {
    "tienkung": {
        "num_dof": 20,
        "joint_vel_slice": slice(20, 40),
        "ee_pos_slice": slice(40, 52),
        "min_cols": 52,
        "joint_names": AMP_JOINT_NAMES_TIENKUNG,
    },
    "f1": {
        "num_dof": 31,
        "joint_vel_slice": slice(31, 62),
        "ee_pos_slice": slice(62, 74),
        "min_cols": 74,
        "joint_names": AMP_JOINT_NAMES_F1,
    },
}

EE_NAMES = ["left_hand", "right_hand", "left_foot", "right_foot"]
FOOT_NAMES = ["left_foot", "right_foot"]


def load_expert_motion(path: Path, robot: str = "tienkung"):
    cfg = ROBOT_CONFIGS[robot]
    with path.open() as f:
        motion = json.load(f)
    frames = np.asarray(motion["Frames"], dtype=np.float64)
    min_cols = cfg["min_cols"]
    if frames.ndim != 2 or frames.shape[1] < min_cols:
        raise ValueError(f"Expected expert frames with at least {min_cols} columns for '{robot}', got shape {frames.shape}.")
    frame_duration = float(motion["FrameDuration"])
    return frames[:, :min_cols], frame_duration, motion


def robust_z(values: np.ndarray, axis=0):
    median = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - median), axis=axis, keepdims=True)
    scale = 1.4826 * mad
    scale = np.maximum(scale, 1.0e-9)
    return np.abs(values - median) / scale


def top_spike_events(scores, values, names, dt, threshold, max_events, frame_offset=0, label=""):
    flat_indices = np.argwhere(scores > threshold)
    events = []
    for index in flat_indices:
        frame = int(index[0]) + frame_offset
        channel = int(index[1])
        events.append(
            {
                "type": label,
                "frame": frame,
                "time_s": frame * dt,
                "name": names[channel],
                "score": float(scores[tuple(index)]),
                "value": float(values[tuple(index)]),
            }
        )
    events.sort(key=lambda item: item["score"], reverse=True)
    return events[:max_events]


def summarize_joint_velocity(joint_vel, joint_acc, dt, threshold, max_events, joint_names):
    joint_acc_abs = np.abs(joint_acc)
    joint_acc_scores = robust_z(joint_acc_abs, axis=0)
    events = top_spike_events(
        joint_acc_scores,
        joint_acc_abs,
        joint_names,
        dt,
        threshold,
        max_events,
        frame_offset=1,
        label="joint_acc",
    )
    p99_acc = np.percentile(joint_acc_abs, 99, axis=0)
    top_joints = np.argsort(p99_acc)[::-1][:5]
    return {
        "max_abs_joint_velocity": float(np.max(np.abs(joint_vel))),
        "max_abs_joint_acceleration": float(np.max(joint_acc_abs)),
        "top_joint_acceleration_p99": [
            {"name": joint_names[idx], "p99_abs_acc": float(p99_acc[idx])} for idx in top_joints
        ],
        "spike_count": int(np.sum(joint_acc_scores > threshold)),
        "events": events,
    }


def summarize_foot_position(foot_pos, dt, threshold, max_events):
    foot_vel = np.diff(foot_pos, axis=0) / dt
    foot_acc = np.diff(foot_vel, axis=0) / dt
    foot_speed = np.linalg.norm(foot_vel, axis=-1)
    foot_acc_norm = np.linalg.norm(foot_acc, axis=-1)
    foot_acc_scores = robust_z(foot_acc_norm, axis=0)
    events = top_spike_events(
        foot_acc_scores,
        foot_acc_norm,
        FOOT_NAMES,
        dt,
        threshold,
        max_events,
        frame_offset=2,
        label="foot_acc",
    )
    return {
        "max_foot_speed": float(np.max(foot_speed)),
        "max_foot_acceleration": float(np.max(foot_acc_norm)),
        "left_foot_z_range": [float(np.min(foot_pos[:, 0, 2])), float(np.max(foot_pos[:, 0, 2]))],
        "right_foot_z_range": [float(np.min(foot_pos[:, 1, 2])), float(np.max(foot_pos[:, 1, 2]))],
        "spike_count": int(np.sum(foot_acc_scores > threshold)),
        "events": events,
    }


def write_events_csv(path: Path, events):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "frame", "time_s", "name", "score", "value"])
        writer.writeheader()
        writer.writerows(events)


def save_plots(output_dir: Path, time, joint_vel, joint_acc, foot_pos, dt, threshold, joint_names):
    import os
    import tempfile

    tmp_dir = tempfile.gettempdir()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tmp_dir) / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", tmp_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 7))
    for idx, name in enumerate(joint_names):
        plt.plot(time, joint_vel[:, idx], linewidth=0.8, alpha=0.8, label=name)
    plt.xlabel("time [s]")
    plt.ylabel("joint velocity [rad/s]")
    plt.title("AMP expert joint velocities")
    plt.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "joint_velocity.png", dpi=160)
    plt.close()

    acc_time = time[1:]
    max_joint_acc = np.max(np.abs(joint_acc), axis=1)
    plt.figure(figsize=(12, 5))
    plt.plot(acc_time, max_joint_acc, linewidth=1.0)
    plt.xlabel("time [s]")
    plt.ylabel("max abs joint acceleration [rad/s^2]")
    plt.title("Joint acceleration envelope")
    plt.tight_layout()
    plt.savefig(output_dir / "joint_acceleration_envelope.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(time, foot_pos[:, 0, 2], label="left_foot_z")
    plt.plot(time, foot_pos[:, 1, 2], label="right_foot_z")
    plt.xlabel("time [s]")
    plt.ylabel("foot z in root frame [m]")
    plt.title("Foot height curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "foot_height.png", dpi=160)
    plt.close()

    foot_vel = np.diff(foot_pos, axis=0) / dt
    foot_acc = np.diff(foot_vel, axis=0) / dt
    foot_speed = np.linalg.norm(foot_vel, axis=-1)
    foot_acc_norm = np.linalg.norm(foot_acc, axis=-1)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time[1:], foot_speed[:, 0], label="left_foot_speed")
    plt.plot(time[1:], foot_speed[:, 1], label="right_foot_speed")
    plt.ylabel("speed [m/s]")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time[2:], foot_acc_norm[:, 0], label="left_foot_acc")
    plt.plot(time[2:], foot_acc_norm[:, 1], label="right_foot_acc")
    plt.axhline(np.median(foot_acc_norm) + threshold * np.std(foot_acc_norm), color="r", linestyle="--", alpha=0.4)
    plt.xlabel("time [s]")
    plt.ylabel("acc [m/s^2]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "foot_speed_acceleration.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Verify AMP expert motion smoothness.")
    parser.add_argument(
        "--robot",
        type=str,
        default="tienkung",
        choices=list(ROBOT_CONFIGS.keys()),
        help="Robot type, determines expert format column layout (default: tienkung).",
    )
    parser.add_argument(
        "--motion",
        type=Path,
        default=Path("legged_lab/envs/tienkung/datasets/motion_amp_expert/walk.txt"),
        help="Path to AMP expert txt/json file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for summary, spike CSV, and plots. Defaults to logs/expert_verification/<motion_stem>.",
    )
    parser.add_argument("--threshold", type=float, default=8.0, help="Robust z-score threshold for spike detection.")
    parser.add_argument("--max_events", type=int, default=20, help="Maximum spike events to print and write.")
    parser.add_argument("--no_plots", action="store_true", help="Skip png plot generation.")
    args = parser.parse_args()

    frames, dt, motion = load_expert_motion(args.motion, args.robot)
    cfg = ROBOT_CONFIGS[args.robot]
    joint_names = cfg["joint_names"]
    output_dir = args.output_dir or Path("logs") / "expert_verification" / args.motion.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    joint_vel = frames[:, cfg["joint_vel_slice"]]
    ee_pos = frames[:, cfg["ee_pos_slice"]].reshape(-1, 4, 3)
    foot_pos = ee_pos[:, 2:4, :]
    joint_acc = np.diff(joint_vel, axis=0) / dt
    time = np.arange(frames.shape[0]) * dt

    joint_summary = summarize_joint_velocity(joint_vel, joint_acc, dt, args.threshold, args.max_events, joint_names)
    foot_summary = summarize_foot_position(foot_pos, dt, args.threshold, args.max_events)
    events = (joint_summary["events"] + foot_summary["events"])[: args.max_events]
    events.sort(key=lambda item: item["score"], reverse=True)

    summary = {
        "motion": str(args.motion),
        "frames": int(frames.shape[0]),
        "frame_duration": dt,
        "duration_s": float((frames.shape[0] - 1) * dt),
        "motion_weight": motion.get("MotionWeight"),
        "threshold": args.threshold,
        "joint_velocity": joint_summary,
        "foot_position": foot_summary,
    }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    write_events_csv(output_dir / "spike_events.csv", events)

    if not args.no_plots:
        save_plots(output_dir, time, joint_vel, joint_acc, foot_pos, dt, args.threshold, joint_names)

    print(f"Motion: {args.motion}")
    print(f"Frames: {frames.shape[0]}, dt: {dt:.6f}s, duration: {summary['duration_s']:.3f}s")
    print(f"Joint acc spikes: {joint_summary['spike_count']}")
    print(f"Foot acc spikes: {foot_summary['spike_count']}")
    print("Top joint acceleration p99:")
    for item in joint_summary["top_joint_acceleration_p99"]:
        print(f"  {item['name']}: {item['p99_abs_acc']:.3f}")
    if events:
        print("Top spike events:")
        for event in events[: args.max_events]:
            print(
                f"  {event['type']} frame={event['frame']} time={event['time_s']:.3f}s "
                f"name={event['name']} score={event['score']:.2f} value={event['value']:.3f}"
            )
    else:
        print("No spike events above threshold.")
    print(f"Saved verification outputs to: {output_dir}")


if __name__ == "__main__":
    main()
