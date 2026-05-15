#!/usr/bin/env python3
"""
interactive_plan.py — Interactive EE targeting with CuRobo + MuJoCo.

Shows the robot in MuJoCo with a red target sphere at the end-effector.
Drag the sphere to a desired position, press SPACE to plan a collision-free
trajectory there. Optionally send the same trajectory to the real robot.

Requires curobo_server.py running in Docker first:
    docker compose run ffw-sg2-planner python3 -u examples/curobo_server.py

Then run on the HOST (GPU-accelerated viewer):
    python3 examples/interactive_plan.py
    python3 examples/interactive_plan.py --robot
    python3 examples/interactive_plan.py --robot --topic /my_controller/joint_trajectory

Controls (in the MuJoCo viewer window):
    Ctrl + drag  →  move the red target sphere to desired EE position
    SPACE        →  send goal to CuRobo server and execute trajectory
    R            →  replay the last trajectory
"""

import argparse
import json
import os
import tempfile
import time
import threading
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
SCENE_XML = REPO_ROOT / "robot/mujoco/scene_ffw_sg2.xml"
GOAL_FILE = REPO_ROOT / "robot/goal.json"
TRAJ_FILE = REPO_ROOT / "robot/traj.npz"

DEFAULT_TOPIC = "/joint_trajectory_controller/joint_trajectory"
PLAN_TIMEOUT  = 30.0   # seconds to wait for curobo_server before giving up
EE_LINK       = "arm_l_link7"

# Home joint config — must match ffw_sg2_left_arm.yml default_joint_position
HOME_JOINTS = {
    "lift_joint":   0.0,
    "arm_l_joint1": 0.0,
    "arm_l_joint2": 0.5,
    "arm_l_joint3": 0.0,
    "arm_l_joint4":-1.2,
    "arm_l_joint5": 0.0,
    "arm_l_joint6": 0.5,
    "arm_l_joint7": 0.0,
}

IDLE      = "idle"
WAITING   = "waiting"
EXECUTING = "executing"


# ── Scene loading ─────────────────────────────────────────────────────────────

def load_scene_with_target(init_pos: np.ndarray) -> mujoco.MjModel:
    """
    Inject a mocap EE-target body into scene_ffw_sg2.xml and load it.
    The mocap body can be dragged in the viewer with Ctrl+drag.
    Writes a temp file alongside the original so that <include> paths resolve.
    """
    x, y, z = init_pos
    mocap_xml = f"""
    <!-- Interactive EE target — move with Ctrl+drag, then press SPACE -->
    <body name="ee_target" mocap="true" pos="{x:.4f} {y:.4f} {z:.4f}">
      <geom type="sphere" size="0.035" rgba="1 0.2 0.2 0.5"
            contype="0" conaffinity="0"/>
      <geom type="box" size="0.005 0.05 0.005" rgba="1 0.2 0.2 1"
            contype="0" conaffinity="0"/>
      <geom type="box" size="0.05 0.005 0.005" rgba="0.2 1 0.2 1"
            contype="0" conaffinity="0"/>
      <geom type="box" size="0.005 0.005 0.05" rgba="0.2 0.2 1 1"
            contype="0" conaffinity="0"/>
    </body>"""

    base_xml = SCENE_XML.read_text()
    merged   = base_xml.replace("</worldbody>", mocap_xml + "\n  </worldbody>")

    # Write to temp file in the same directory so relative includes resolve
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=SCENE_XML.parent, delete=False)
    tmp.write(merged)
    tmp.close()
    try:
        model = mujoco.MjModel.from_xml_path(tmp.name)
    finally:
        os.unlink(tmp.name)
    return model


def get_ee_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World-frame position of arm_l_link7 after forward kinematics."""
    mujoco.mj_forward(model, data)
    body_id = model.body(EE_LINK).id
    return data.xpos[body_id].copy()


# ── Goal file I/O ─────────────────────────────────────────────────────────────

def send_goal(position: list, req_id: int):
    goal = {"position": position, "id": req_id, "status": "pending"}
    tmp  = GOAL_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(goal))
    tmp.rename(GOAL_FILE)


def read_goal_status() -> str | None:
    try:
        return json.loads(GOAL_FILE.read_text()).get("status")
    except Exception:
        return None


# ── ROS 2 publishing ──────────────────────────────────────────────────────────

def publish_ros2(joint_names, positions, dt, topic):
    try:
        import rclpy
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration

        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("ffw_interactive_pub")
        pub  = node.create_publisher(JointTrajectory, topic, 10)
        time.sleep(0.3)   # wait for subscriber discovery

        msg              = JointTrajectory()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.joint_names  = list(joint_names)

        for t, pos in enumerate(positions):
            pt          = JointTrajectoryPoint()
            pt.positions = [float(v) for v in pos]
            ns = int((t + 1) * dt * 1e9)
            pt.time_from_start = Duration(
                sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
            msg.points.append(pt)

        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.2)
        node.destroy_node()
        print(f"[ros2] Published {len(positions)} waypoints → {topic}")
    except ImportError:
        print("[ros2] rclpy not found — skipping robot publish")
    except Exception as e:
        print(f"[ros2] Error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Interactive CuRobo + MuJoCo planner")
    p.add_argument("--robot",  action="store_true",
                   help="Also send trajectory to real robot via ROS 2")
    p.add_argument("--topic",  default=DEFAULT_TOPIC,
                   help=f"ROS 2 controller topic (default: {DEFAULT_TOPIC})")
    args = p.parse_args()

    # ── Find home EE position for initial target placement ─────────────────
    _m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    _d = mujoco.MjData(_m)
    for i in range(_m.nu):
        name = _m.actuator(i).name
        if name in HOME_JOINTS:
            _d.ctrl[i] = HOME_JOINTS[name]
    home_ee = get_ee_pos(_m, _d)
    del _m, _d

    # ── Load interactive scene ─────────────────────────────────────────────
    model = load_scene_with_target(home_ee)
    data  = mujoco.MjData(model)

    ctrl_idx = {model.actuator(i).name: i for i in range(model.nu)}

    # ee_target is the only mocap body — index 0
    mocap_id = 0

    # Initialise robot at home config
    for name, val in HOME_JOINTS.items():
        if name in ctrl_idx:
            data.ctrl[ctrl_idx[name]] = val

    # ── State machine ──────────────────────────────────────────────────────
    state      = IDLE
    req_id     = 0
    traj       = None    # last successful trajectory (dict)
    traj_start = 0.0
    wait_start = 0.0
    poll_tick  = 0

    plan_flag   = threading.Event()
    replay_flag = threading.Event()

    def key_callback(keycode):
        if keycode == 32:          # SPACE
            plan_flag.set()
        elif keycode == ord("R"):  # R
            replay_flag.set()

    print("\n[info] Controls:")
    print("  Ctrl+drag in viewer  →  drag the red sphere to desired EE position")
    print("  SPACE                →  plan + execute trajectory to the sphere")
    print("  R                    →  replay last trajectory")
    if args.robot:
        print(f"  --robot active       →  will publish to {args.topic}")
    print("\n[info] curobo_server.py must be running in Docker.\n")

    with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback) as viewer:

        viewer.cam.distance  = 3.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 135

        mujoco_dt = model.opt.timestep

        while viewer.is_running():
            step_start = time.perf_counter()

            # ── State transitions ──────────────────────────────────────────
            if plan_flag.is_set() and state == IDLE:
                plan_flag.clear()
                target = data.mocap_pos[mocap_id].tolist()
                req_id += 1
                send_goal(target, req_id)
                state      = WAITING
                wait_start = time.perf_counter()
                print(f"[plan] Goal → {[f'{v:.3f}' for v in target]}  "
                      f"(waiting for curobo_server...)")

            if replay_flag.is_set() and traj is not None and state == IDLE:
                replay_flag.clear()
                traj_start = time.perf_counter()
                state      = EXECUTING
                print(f"[plan] Replaying {traj['positions'].shape[0]} waypoints")

            if state == WAITING:
                poll_tick += 1
                if poll_tick % 25 == 0:   # check every ~50 ms
                    status = read_goal_status()
                    if status == "done":
                        npz  = np.load(TRAJ_FILE, allow_pickle=True)
                        traj = {
                            "joint_names": list(npz["joint_names"]),
                            "positions":   npz["positions"],
                            "dt":          float(npz["dt"]),
                        }
                        T = traj["positions"].shape[0]
                        print(f"[plan] Executing {T} waypoints "
                              f"({T * traj['dt']:.2f}s)")
                        if args.robot:
                            threading.Thread(
                                target=publish_ros2,
                                args=(traj["joint_names"], traj["positions"],
                                      traj["dt"], args.topic),
                                daemon=True,
                            ).start()
                        traj_start = time.perf_counter()
                        state      = EXECUTING

                    elif status == "failed":
                        state = IDLE
                        print("[plan] Server: goal unreachable or in collision. "
                              "Try a different target.")

                    elif time.perf_counter() - wait_start > PLAN_TIMEOUT:
                        state = IDLE
                        print(f"[plan] Timeout ({PLAN_TIMEOUT}s). "
                              "Is curobo_server.py running in Docker?")

            if state == EXECUTING and traj is not None:
                T       = traj["positions"].shape[0]
                elapsed = time.perf_counter() - traj_start
                t       = min(int(elapsed / traj["dt"]), T - 1)

                for j, name in enumerate(traj["joint_names"]):
                    if name in ctrl_idx:
                        data.ctrl[ctrl_idx[name]] = traj["positions"][t, j]

                if elapsed >= T * traj["dt"]:
                    state = IDLE
                    # Move target to actual EE so next drag starts there
                    data.mocap_pos[mocap_id] = get_ee_pos(model, data)
                    print("[plan] Done. Drag the sphere and press SPACE to plan again.")

            # ── Physics + render ───────────────────────────────────────────
            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time pacing
            elapsed   = time.perf_counter() - step_start
            remaining = mujoco_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
