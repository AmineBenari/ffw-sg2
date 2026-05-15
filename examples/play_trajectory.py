#!/usr/bin/env python3
"""
play_trajectory.py — Replay a CuRobo trajectory in MuJoCo, optionally on the real robot.

Loads robot/traj.npz (produced by curobo_motion.py).

Run on the HOST (native OpenGL → GPU-accelerated viewer):
    python3 examples/play_trajectory.py              # sim only
    python3 examples/play_trajectory.py --robot      # sim + real robot

Options:
    --robot              Also publish the trajectory to the real robot via ROS 2
    --topic TOPIC        ROS 2 JointTrajectoryController topic
                         (default: /joint_trajectory_controller/joint_trajectory)
    --loop               Replay trajectory in a loop (sim only)
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
SCENE_XML = REPO_ROOT / "robot/mujoco/scene_ffw_sg2.xml"
TRAJ_FILE = REPO_ROOT / "robot/traj.npz"

DEFAULT_TOPIC = "/joint_trajectory_controller/joint_trajectory"


# ── ROS 2 publisher ───────────────────────────────────────────────────────

def publish_trajectory_ros2(joint_names, positions, dt, topic):
    """
    Publish the full trajectory as a trajectory_msgs/JointTrajectory message.
    The JointTrajectoryController on the robot will interpolate and execute it.

    Returns the rclpy node (caller must call node.destroy_node() + rclpy.shutdown()).
    """
    import rclpy
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration

    rclpy.init()
    node = rclpy.create_node("ffw_trajectory_publisher")
    pub  = node.create_publisher(JointTrajectory, topic, qos_profile=10)

    # Give the publisher time to connect to subscribers before sending
    time.sleep(0.5)

    msg = JointTrajectory()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.joint_names  = list(joint_names)

    T = positions.shape[0]
    for t in range(T):
        pt = JointTrajectoryPoint()
        pt.positions = [float(positions[t, j]) for j in range(len(joint_names))]
        total_ns     = int((t + 1) * dt * 1e9)
        pt.time_from_start = Duration(
            sec=total_ns // 1_000_000_000,
            nanosec=total_ns % 1_000_000_000,
        )
        msg.points.append(pt)

    pub.publish(msg)
    # Spin briefly to let the message flush
    rclpy.spin_once(node, timeout_sec=0.2)

    print(f"[ros2] Published {T} waypoints → {topic}")
    return node


# ── MuJoCo playback ───────────────────────────────────────────────────────

def play(joint_names, positions, curobo_dt, loop=False):
    model     = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data      = mujoco.MjData(model)
    mujoco_dt = model.opt.timestep

    # Map trajectory joint names → actuator ctrl indices
    ctrl_idx: dict[str, int] = {}
    for i in range(model.nu):
        name = model.actuator(i).name
        if name in joint_names:
            ctrl_idx[name] = i

    print(f"[mujoco] Mapped {len(ctrl_idx)}/{len(joint_names)} joints to actuators")

    steps_per_wpt = max(1, round(curobo_dt / mujoco_dt))
    T = positions.shape[0]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = 3.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 135

        while True:
            print(f"[mujoco] Playing {T} waypoints ({T * curobo_dt:.2f}s)")
            wall_start = time.perf_counter()

            for t in range(T):
                if not viewer.is_running():
                    return

                for j, name in enumerate(joint_names):
                    if name in ctrl_idx:
                        data.ctrl[ctrl_idx[name]] = positions[t, j]

                for _ in range(steps_per_wpt):
                    mujoco.mj_step(model, data)

                viewer.sync()

                # Real-time pacing
                slack = (t + 1) * curobo_dt - (time.perf_counter() - wall_start)
                if slack > 0:
                    time.sleep(slack)

            if not loop:
                break

        print("[mujoco] Holding final pose.")
        while viewer.is_running():
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = mujoco_dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Replay CuRobo trajectory in MuJoCo")
    p.add_argument("--robot",  action="store_true",
                   help="Also send trajectory to real robot via ROS 2")
    p.add_argument("--topic",  default=DEFAULT_TOPIC,
                   help=f"ROS 2 controller topic (default: {DEFAULT_TOPIC})")
    p.add_argument("--loop",   action="store_true",
                   help="Replay trajectory in a loop (sim only)")
    args = p.parse_args()

    # Load trajectory
    npz          = np.load(TRAJ_FILE, allow_pickle=True)
    joint_names  = list(npz["joint_names"])
    positions    = npz["positions"]      # (T, dof)
    curobo_dt    = float(npz["dt"])
    print(f"[traj] {positions.shape[0]} waypoints × {curobo_dt*1000:.1f} ms  "
          f"= {positions.shape[0]*curobo_dt:.2f}s")
    print(f"[traj] joints: {joint_names}")

    # Publish to robot first (before MuJoCo starts) so they stay roughly in sync
    ros_node = None
    if args.robot:
        try:
            ros_node = publish_trajectory_ros2(joint_names, positions, curobo_dt, args.topic)
        except ImportError:
            print("[ros2] ERROR: rclpy not found. Install ROS 2 Jazzy on the host or "
                  "run this script from inside the Docker container.")
            raise

    try:
        play(joint_names, positions, curobo_dt, loop=args.loop)
    finally:
        if ros_node is not None:
            import rclpy
            ros_node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
