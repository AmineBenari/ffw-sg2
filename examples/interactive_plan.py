#!/usr/bin/env python3
"""
interactive_plan.py — Interactive EE targeting with live reachability preview.

Move the target sphere:
  Mouse    → Ctrl + left-drag  (free XY in view plane)
             Ctrl + right-drag (depth / Z)
  X / Y / Z  → lock mouse drag to that axis (press again to release)
  Esc        → release axis lock
  Arrow keys → camera-relative move (right/left = strafe, up/down = forward/back)
  Page Up/Dn → world Z
  T          → cycle active arm (when --arm both)
  + / -      → raise / lower lift

Sphere colour shows reachability in real time:
  Grey    — not yet evaluated (just moved)
  Yellow  — CuRobo is planning
  Green   — reachable   (ghost arm animates the planned trajectory in a loop)
  Red     — out of reach

Press SPACE to execute the last successful plan (active arm).
Press H     to reset all arms and spheres to home.
Press R     to replay the last trajectory (active arm).

Orientation control (only with --orientation flag):
  I / K   — pitch the target frame up / down  (world Y rotation)
  J / L   — yaw  the target frame left / right (world Z rotation)
  U / O   — roll the target frame ccw / cw     (world X rotation)
  Sphere axis bars show the target orientation (red=X, green=Y, blue=Z).

Requires curobo_server.py running in Docker first:
    # Single arm:
    docker compose -f docker/docker-compose.yml run --name ffw_server \\
        -d ffw-sg2-planner python3 -u examples/curobo_server.py [--arm right] [--orientation]
    # Both arms simultaneously (two containers):
    docker compose -f docker/docker-compose.yml run --name ffw_server_l \\
        -d ffw-sg2-planner python3 -u examples/curobo_server.py --arm left
    docker compose -f docker/docker-compose.yml run --name ffw_server_r \\
        -d ffw-sg2-planner python3 -u examples/curobo_server.py --arm right

Then run on the host:
    python3 examples/interactive_plan.py               # both arms (default)
    python3 examples/interactive_plan.py --arm left    # left arm only
    python3 examples/interactive_plan.py --arm right   # right arm only
    python3 examples/interactive_plan.py --orientation    # also start server(s) with --orientation
    python3 examples/interactive_plan.py --robot          # also send via FollowJointTrajectory
"""

import argparse
import json
import os
import tempfile
import time
import threading
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT  = Path(__file__).parent.parent
SCENE_XML  = REPO_ROOT / "robot/mujoco/scene_ffw_sg2.xml"

PLAN_TIMEOUT = 30.0
_VISUAL_GROUP = 2

# ── Lift configuration ────────────────────────────────────────────────────────

LIFT_JOINT = "lift_joint"
LIFT_RANGE = (-0.5, 0.0)
# arm_base_link offset from base_link (XY fixed, Z = 1.4316 + lift_val)
LIFT_ARM_BASE_OFFSET = np.array([0.0055, 0.0, 1.4316])
# All mesh bodies that move with the lift (ghost shows full upper body)
LIFT_GHOST_LINKS = [
    "arm_base_link",
    "head_link1", "head_link2",
    "arm_l_link1", "arm_l_link2", "arm_l_link3", "arm_l_link4",
    "arm_l_link5", "arm_l_link6", "arm_l_link7",
    "gripper_l_r1", "gripper_l_l1", "gripper_l_r2", "gripper_l_l2",
    "arm_r_link1", "arm_r_link2", "arm_r_link3", "arm_r_link4",
    "arm_r_link5", "arm_r_link6", "arm_r_link7",
    "gripper_r_r1", "gripper_r_l1", "gripper_r_r2", "gripper_r_l2",
]
# Smooth execution: seconds to travel full range
LIFT_EXEC_SPEED = 0.15   # m/s

# ── Per-arm configuration ─────────────────────────────────────────────────────

ARM_CONFIGS = {
    "left": {
        "ee_site":      "arm_l_tcp",
        "plan_joints":  ["arm_l_joint1", "arm_l_joint2", "arm_l_joint3",
                         "arm_l_joint4", "arm_l_joint5", "arm_l_joint6", "arm_l_joint7"],
        "ghost_links":  ["arm_l_link1", "arm_l_link2", "arm_l_link3",
                         "arm_l_link4", "arm_l_link5", "arm_l_link6", "arm_l_link7",
                         "gripper_l_r1", "gripper_l_l1", "gripper_l_r2", "gripper_l_l2"],
        "action_topic": "/arm_l_controller/follow_joint_trajectory",
        "gripper_joint": "gripper_l_joint1",
        "home_joints": {
            "lift_joint": 0.0,
            "arm_l_joint1": 0.0, "arm_l_joint2": 0.0, "arm_l_joint3": 0.0,
            "arm_l_joint4": 0.0, "arm_l_joint5": 0.0, "arm_l_joint6": 0.0,
            "arm_l_joint7": 0.0,
        },
    },
    "right": {
        "ee_site":      "arm_r_tcp",
        "plan_joints":  ["arm_r_joint1", "arm_r_joint2", "arm_r_joint3",
                         "arm_r_joint4", "arm_r_joint5", "arm_r_joint6", "arm_r_joint7"],
        "ghost_links":  ["arm_r_link1", "arm_r_link2", "arm_r_link3",
                         "arm_r_link4", "arm_r_link5", "arm_r_link6", "arm_r_link7",
                         "gripper_r_r1", "gripper_r_l1", "gripper_r_r2", "gripper_r_l2"],
        "action_topic": "/arm_r_controller/follow_joint_trajectory",
        "gripper_joint": "gripper_r_joint1",
        "home_joints": {
            "lift_joint": 0.0,
            "arm_r_joint1": 0.0, "arm_r_joint2": 0.0, "arm_r_joint3": 0.0,
            "arm_r_joint4": 0.0, "arm_r_joint5": 0.0, "arm_r_joint6": 0.0,
            "arm_r_joint7": 0.0,
        },
    },
}

# ── Sphere RGBA states ────────────────────────────────────────────────────────

RGBA_IDLE     = np.array([0.85, 0.85, 0.85, 0.55], dtype=np.float32)
RGBA_PLANNING = np.array([1.00, 0.85, 0.00, 0.70], dtype=np.float32)
RGBA_OK       = np.array([0.10, 0.90, 0.10, 0.60], dtype=np.float32)
RGBA_FAIL     = np.array([1.00, 0.15, 0.15, 0.70], dtype=np.float32)

RGBA_GHOST_OK   = np.array([0.15, 0.90, 0.15, 0.35], dtype=np.float32)
RGBA_GHOST_FAIL = np.array([1.00, 0.20, 0.20, 0.35], dtype=np.float32)

IDLE      = "idle"
WAITING   = "waiting"
EXECUTING = "executing"

# Seconds of sphere stillness before auto-planning fires
PLAN_DEBOUNCE = 0.40

# Orientation rotation step per keypress (5 degrees)
ROT_STEP = np.deg2rad(5.0)

# GLFW key codes
KEY_SPACE    = 32
KEY_ESC      = 256
KEY_TAB      = 258
KEY_R        = ord("R");  KEY_H = ord("H")
KEY_LEFT     = 263;       KEY_RIGHT  = 262
KEY_UP       = 265;       KEY_DOWN   = 264
KEY_PAGE_UP  = 266;       KEY_PAGE_DN = 267
KEY_LBRACKET = ord("[");  KEY_RBRACKET = ord("]")
# Axis constraint keys
KEY_X = ord("X");  KEY_Y = ord("Y");  KEY_ZKEY = ord("Z")
# Orientation keys (I/K = pitch, J/L = yaw, U/O = roll)
KEY_I = ord("I");  KEY_K = ord("K")
KEY_J = ord("J");  KEY_L = ord("L")
KEY_U = ord("U");  KEY_O = ord("O")
# Cycle active arm (T avoids MuJoCo's built-in Tab = side-panel toggle)
KEY_T = ord("T")
# Lift keys: - to lower, = to raise (= is the +/- key without shift on most keyboards)
KEY_LIFT_UP   = ord("=")   # same physical key as + (no shift needed)
KEY_LIFT_DOWN = ord("-")

KEY_W = ord("W");  KEY_S = ord("S")   # drive forward / backward
KEY_A = ord("A");  KEY_D = ord("D")   # turn left / right

DRIVE_LIN_VEL  = 0.4    # m/s  (matches real teleop)
DRIVE_ANG_VEL  = 0.8    # rad/s (matches real teleop)
STEER_RATE     = 8.0    # rad/s — visually smooth in sim (real: 100 rad/s but sub-frame so jumpy)
STEER_ALIGN_THRESH      = 1.0   # rad — matches steering_alignment_angle_error_threshold in real config
STEER_ALIGN_THRESH_IDLE = 0.01  # rad — matches steering_alignment_start_angle_error_threshold (now reachable with kinematic steering)
WHEEL_RADIUS   = 0.09   # m — matches cylinder geom size in ffw_sg2.xml (real robot: 0.0825)
WHEEL_SPEED_LIMIT = 50.0  # rad/s — matches module_wheel_speed_limit in real config

# Direction-reversal FSM constants (matches real swerve_drive_controller.cpp)
REVERSAL_NORMAL   = 0
REVERSAL_DECEL    = 1
REVERSAL_STEERING = 2
REVERSAL_ACCEL    = 3
REVERSAL_DECEL_RATE  = 7.0   # scale/s — kReversalDecelRate in real controller
REVERSAL_ACCEL_RATE  = 5.0   # scale/s — kReversalAccelRate in real controller
REVERSAL_THRESHOLD   = 0.05  # stop decel at this scale
REVERSAL_STEER_TOL   = 0.1   # rad — steering tolerance to leave STEERING phase

# Swerve modules: (name, module_x, module_y, steer_actuator, drive_actuator, angle_offset)
# angle_offset: per-module calibration (subtracted from IK target) — matches
# module_angle_offsets in ffw_sg2_follower_ai_hardware_controller.yaml (all 0.0).
SWERVE_MODULES = (
    ("left",  0.1371,  0.2554, "left_wheel_steer",  "left_wheel_drive",  0.0),
    ("right", 0.1371, -0.2554, "right_wheel_steer", "right_wheel_drive", 0.0),
    ("rear", -0.2899,  0.0,    "rear_wheel_steer",  "rear_wheel_drive",  0.0),
)

# Axis-constraint direction vectors
_AXIS_VEC = {
    'x': np.array([1.0, 0.0, 0.0]),
    'y': np.array([0.0, 1.0, 0.0]),
    'z': np.array([0.0, 0.0, 1.0]),
}


# ── Quaternion helpers ────────────────────────────────────────────────────────

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two [w, x, y, z] unit quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Build a [w, x, y, z] quaternion for a rotation around `axis` by `angle` rad."""
    half = angle / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


def _rotate_world(q: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate quaternion q by `angle` around world-frame `axis`, return normalised result."""
    dq = _axis_angle_quat(axis, angle)
    result = _quat_mul(dq, q)   # world-frame: pre-multiply
    return result / np.linalg.norm(result)


# ── Scene helpers ─────────────────────────────────────────────────────────────

def load_scene_with_targets(arms_init: list,
                            lift_pos: np.ndarray | None = None) -> mujoco.MjModel:
    """Create a scene with one mocap target body per arm and an optional lift target.

    arms_init: list of (arm_name, init_pos_np)
    lift_pos:  initial world position of the lift target sphere (or None to omit)
    """
    mocap_xml = ""
    for arm, pos in arms_init:
        x, y, z = pos
        rgba_str = " ".join(f"{v:.3f}" for v in RGBA_IDLE)
        mocap_xml += f"""
    <!-- {arm} arm target — Ctrl+drag -->
    <body name="ee_target_{arm}" mocap="true" pos="{x:.4f} {y:.4f} {z:.4f}">
      <geom name="ee_target_sphere_{arm}" type="sphere" size="0.035"
            rgba="{rgba_str}" contype="0" conaffinity="0"/>
      <!-- X axis — red shaft + sphere tip -->
      <geom type="box"    size="0.040 0.004 0.004" pos="0.040 0 0"
            rgba="1.0 0.15 0.15 0.9" contype="0" conaffinity="0"/>
      <geom type="sphere" size="0.011"             pos="0.082 0 0"
            rgba="1.0 0.15 0.15 0.9" contype="0" conaffinity="0"/>
      <!-- Y axis — green shaft + sphere tip -->
      <geom type="box"    size="0.004 0.040 0.004" pos="0 0.040 0"
            rgba="0.15 1.0 0.15 0.9" contype="0" conaffinity="0"/>
      <geom type="sphere" size="0.011"             pos="0 0.082 0"
            rgba="0.15 1.0 0.15 0.9" contype="0" conaffinity="0"/>
      <!-- Z axis — blue shaft + sphere tip -->
      <geom type="box"    size="0.004 0.004 0.040" pos="0 0 0.040"
            rgba="0.15 0.15 1.0 0.9" contype="0" conaffinity="0"/>
      <geom type="sphere" size="0.011"             pos="0 0 0.082"
            rgba="0.15 0.15 1.0 0.9" contype="0" conaffinity="0"/>
    </body>"""
    if lift_pos is not None:
        x, y, z = lift_pos
        rgba_str = " ".join(f"{v:.3f}" for v in RGBA_IDLE)
        mocap_xml += f"""
    <!-- lift target — drag Z to set lift height -->
    <body name="lift_target" mocap="true" pos="{x:.4f} {y:.4f} {z:.4f}">
      <geom name="lift_target_sphere" type="sphere" size="0.040"
            rgba="{rgba_str}" contype="0" conaffinity="0"/>
      <!-- Z-only indicator bar (cyan) -->
      <geom type="box" size="0.004 0.004 0.060" pos="0 0 0.060"
            rgba="0.0 0.85 0.85 0.9" contype="0" conaffinity="0"/>
      <geom type="sphere" size="0.013" pos="0 0 0.122"
            rgba="0.0 0.85 0.85 0.9" contype="0" conaffinity="0"/>
    </body>"""
    base_xml = SCENE_XML.read_text()
    merged   = base_xml.replace("</worldbody>", mocap_xml + "\n  </worldbody>")
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=SCENE_XML.parent, delete=False)
    tmp.write(merged); tmp.close()
    try:
        model = mujoco.MjModel.from_xml_path(tmp.name)
    finally:
        os.unlink(tmp.name)
    return model


def get_ee_pos(model, data, ee_site: str) -> np.ndarray:
    mujoco.mj_forward(model, data)
    return data.site_xpos[model.site(ee_site).id].copy()


# ── File-based planning IPC ───────────────────────────────────────────────────

def send_goal(position: list, req_id: int, joint_state: dict,
              orientation: list | None, goal_file: Path):
    goal: dict = {"position": position, "joint_state": joint_state,
                  "id": req_id, "status": "pending"}
    if orientation is not None:
        goal["orientation"] = orientation
    tmp = goal_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(goal))
    tmp.rename(goal_file)


def read_goal(goal_file: Path) -> dict | None:
    try:
        return json.loads(goal_file.read_text())
    except Exception:
        return None


# ── Ghost arm rendering ───────────────────────────────────────────────────────

def set_ghost(viewer, model, ghost_data, data,
              ghost_joints, joint_names, ok: bool,
              ghost_links: tuple | None = None,
              clear_scene: bool = True,
              pre_fk: bool = False):
    """Render the arm at ghost_joints using the actual visual meshes.

    clear_scene=False lets a second arm's ghost be appended without clearing.
    pre_fk=True: ghost_data.qpos is already set by the caller; skip qpos copy,
                 just run mj_forward and render (used for lift ghost).
    """
    cache_key = (id(model), ghost_links)
    if not hasattr(set_ghost, '_geom_cache'):
        set_ghost._geom_cache = {}
    if cache_key not in set_ghost._geom_cache:
        arm_body_ids = frozenset(model.body(n).id for n in ghost_links)
        mesh_t = int(mujoco.mjtGeom.mjGEOM_MESH)
        cat    = int(mujoco.mjtCatBit.mjCAT_DYNAMIC)
        arm_geoms = []
        for gi in range(model.ngeom):
            if model.geom_bodyid[gi] not in arm_body_ids:
                continue
            if model.geom_type[gi] != mesh_t:
                continue
            grp = model.geom_group[gi]
            if grp not in (0, _VISUAL_GROUP):
                continue
            raw_did = int(model.geom_dataid[gi])
            if raw_did < 0:
                continue
            arm_geoms.append((gi, 2 * raw_did,
                              model.geom_size[gi].astype(np.float64).copy()))
        set_ghost._geom_cache[cache_key] = (arm_geoms, mesh_t, cat)

    arm_geoms, mesh_t, cat = set_ghost._geom_cache[cache_key]

    if ghost_joints is not None:
        ghost_data.qpos[:] = data.qpos[:]
        for j, name in enumerate(joint_names):
            try:
                ghost_data.qpos[model.jnt_qposadr[model.joint(name).id]] = (
                    np.asarray(ghost_joints[j]).item())
            except Exception:
                pass
        mujoco.mj_forward(model, ghost_data)
    elif pre_fk:
        mujoco.mj_forward(model, ghost_data)

    rgba = (RGBA_GHOST_OK if ok else RGBA_GHOST_FAIL).astype(np.float32)

    with viewer.lock():
        scn = viewer.user_scn
        if clear_scene:
            scn.ngeom = 0
        if ghost_joints is None and not pre_fk:
            return
        n = scn.ngeom
        for (gi, did, sz) in arm_geoms:
            if n >= scn.maxgeom:
                break
            dst = scn.geoms[n]
            mujoco.mjv_initGeom(
                dst, mesh_t, sz,
                ghost_data.geom_xpos[gi],
                ghost_data.geom_xmat[gi],
                rgba,
            )
            dst.dataid      = did
            dst.category    = cat
            dst.transparent = 1
            dst.objtype     = int(mujoco.mjtObj.mjOBJ_GEOM)
            dst.objid       = gi
            dst.segid       = -1
            n += 1
        scn.ngeom = n


# ── ROS 2 publish ─────────────────────────────────────────────────────────────

def publish_ros2(joint_names, positions, dt, action_topic, gripper_joint,
                 ready_event: threading.Event | None = None):
    """Send trajectory via FollowJointTrajectory action.

    Sets ready_event once the controller accepts the goal so the caller can
    synchronise MuJoCo playback with hardware execution.
    """
    try:
        import rclpy
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        from builtin_interfaces.msg import Duration

        if not rclpy.ok():
            rclpy.init()

        node   = rclpy.create_node("ffw_interactive_pub")
        client = ActionClient(node, FollowJointTrajectory, action_topic)

        # Filter to arm joints only and append gripper held at 0
        arm_prefix = "arm_l_" if "arm_l" in str(joint_names) else "arm_r_"
        arm_names  = [n for n in joint_names if n.startswith(arm_prefix)]
        arm_idxs   = [list(joint_names).index(n) for n in arm_names]
        out_names  = arm_names + [gripper_joint]

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = out_names
        for t, pos in enumerate(positions):
            pt = JointTrajectoryPoint()
            pt.positions = [float(pos[i]) for i in arm_idxs] + [0.0]
            ns = int((t + 1) * dt * 1e9)
            pt.time_from_start = Duration(
                sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
            goal_msg.trajectory.points.append(pt)

        if not client.wait_for_server(timeout_sec=5.0):
            print(f"[ros2] Action server not available: {action_topic}")
            node.destroy_node()
            return

        future = client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        handle = future.result()
        if not handle.accepted:
            print("[ros2] Goal rejected by controller")
            node.destroy_node()
            return

        # Signal that hardware has accepted — MuJoCo playback can now start
        if ready_event is not None:
            ready_event.set()

        print(f"[ros2] Sent {len(positions)} waypoints → {action_topic}")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future, timeout_sec=60.0)
        result = result_future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            print("[ros2] Trajectory completed successfully")
        else:
            print(f"[ros2] Trajectory error: code={result.error_code}  {result.error_string}")
        node.destroy_node()

    except ImportError:
        print("[ros2] rclpy / control_msgs not found — skipping robot publish")
        if ready_event is not None:
            ready_event.set()   # unblock MuJoCo so sim still runs
    except Exception as e:
        print(f"[ros2] Error: {e}")
        if ready_event is not None:
            ready_event.set()   # unblock on error too


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["left", "right", "both"], default="both",
                   help="Which arm(s) to plan for — must match running curobo_server.py instance(s)")
    p.add_argument("--orientation", action="store_true",
                   help="Enable 6DOF goal (position + orientation). "
                        "Server must also be started with --orientation.")
    p.add_argument("--robot", action="store_true",
                   help="Also send trajectory to real robot via FollowJointTrajectory action")
    p.add_argument("--topic", default=None,
                   help="Override the default action topic (single-arm mode only)")
    args = p.parse_args()

    arms = ["left", "right"] if args.arm == "both" else [args.arm]

    print(f"\n[info] Arm(s): {', '.join(arms)}")
    print(f"[info] Orientation mode: {'6DOF' if args.orientation else 'position-only'}")
    print(f"[info] Robot publish: {'yes' if args.robot else 'no'}")
    print("\n[info] Controls (click viewer window first):")
    print("  Mouse          → Ctrl + left-drag (free XY),  Ctrl + right-drag (depth)")
    print("  X / Y / Z      → lock Ctrl+drag to that axis  (press again to release)")
    print("  Esc            → release axis lock")
    print("  Arrow keys     → camera-relative move (right/left = strafe, up/down = fwd/back)")
    print("  Page Up/Down   → world Z")
    print("  [ / ]          → halve / double keyboard step size")
    if len(arms) > 1:
        print("  T              → cycle active arm for arrow-key control")
    print("  + / -          → raise / lower lift (always active)")
    print("  W / S          → drive base forward / backward (hold)")
    print("  A / D          → turn base left / right (hold)")
    if args.orientation:
        print("  I / K          → pitch target frame up / down")
        print("  J / L          → yaw  target frame left / right")
        print("  U / O          → roll target frame ccw / cw")
    print("  SPACE          → execute last successful plan (active arm)")
    print("  R              → replay last trajectory (active arm)")
    print("  H              → reset all arms and spheres to home")
    print()
    print("  Sphere:  grey=idle  yellow=planning  green=ok (ghost animates path)  red=unreachable\n")

    # Compute MuJoCo home EE position for each arm
    _m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    _d = mujoco.MjData(_m)
    mujoco.mj_forward(_m, _d)
    init_ee = {arm: get_ee_pos(_m, _d, ARM_CONFIGS[arm]["ee_site"]) for arm in arms}
    del _m, _d

    # Build per-arm state
    arm_states = {}
    for arm in arms:
        arm_states[arm] = {
            "cfg":              ARM_CONFIGS[arm],
            "goal_file":        REPO_ROOT / f"robot/goal_{arm}.json",
            "traj_file":        REPO_ROOT / f"robot/traj_{arm}.npz",
            "server_info":      REPO_ROOT / f"robot/server_info_{arm}.json",
            "frame_offset":     np.zeros(3),
            "state":            IDLE,
            "req_id":           0,
            "traj":             None,
            "traj_start":       0.0,
            "wait_start":       0.0,
            "ghost_traj":       None,
            "ghost_traj_dt":    0.05,
            "ghost_anim_t0":    0.0,
            "ghost_joint_names": None,
            "ghost_ok":         False,
            "last_planned_pos":  init_ee[arm].copy(),
            "last_planned_base": np.zeros(3),   # last_planned_pos in robot base frame
            "last_planned_quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "prev_sphere_pos":   init_ee[arm].copy(),
            "prev_sphere_quat":  np.array([1.0, 0.0, 0.0, 0.0]),
            "prev_tgt_base":    None,   # filled after model load + FK
            "last_move_time":   0.0,
            # Target position stored in robot-base frame to follow base without drift
            "tgt_base":         None,   # filled after model load + FK
            "last_set_world":   None,   # world pos we last wrote to mocap (to detect drag)
            # Filled after model load:
            "mocap_id":         None,
            "sphere_geom_id":   None,
            "ghost_data":       None,
        }

    # Initialise req_id from existing goal files so we never send an id that
    # the already-running server already processed (which it silently skips).
    for arm in arms:
        gf = arm_states[arm]["goal_file"]
        try:
            existing = json.loads(gf.read_text())
            arm_states[arm]["req_id"] = int(existing.get("id", 0))
        except Exception:
            pass

    # Try to read server_info once now — if the server isn't ready yet we'll
    # pick it up lazily in the main loop (see _try_load_frame_offset below).
    _server_info_pending = set(arms)
    for arm in arms:
        as_ = arm_states[arm]
        try:
            info = json.loads(as_["server_info"].read_text())
            if info.get("arm") == arm:
                curobo_home = np.array(info["ee_home_curobo"])
                as_["frame_offset"] = init_ee[arm] - curobo_home
                print(f"[info] [{arm}] Frame offset (MuJoCo−CuRobo): "
                      f"{as_['frame_offset'][0]:+.4f}  "
                      f"{as_['frame_offset'][1]:+.4f}  "
                      f"{as_['frame_offset'][2]:+.4f}")
                _server_info_pending.discard(arm)
        except Exception:
            pass
    if _server_info_pending:
        print(f"[info] Server info not ready for arm(s): "
              f"{', '.join(_server_info_pending)} — will retry in the main loop.")

    # Compute lift home sphere position (arm_base_link at lift_joint=0)
    _lift_home_base = LIFT_ARM_BASE_OFFSET.copy()  # [0.0055, 0, 1.4316] in base frame

    # Load scene with one mocap target per arm + lift
    model = load_scene_with_targets(
        [(arm, init_ee[arm]) for arm in arms],
        lift_pos=_lift_home_base,   # will be corrected after FK below
    )
    data  = mujoco.MjData(model)
    ctrl_idx = {model.actuator(i).name: i for i in range(model.nu)}
    mujoco.mj_forward(model, data)

    # Fill in per-arm model-specific IDs now that the model is loaded
    for arm, as_ in arm_states.items():
        body_id = model.body(f"ee_target_{arm}").id
        as_["mocap_id"]       = model.body_mocapid[body_id]
        as_["sphere_geom_id"] = model.geom(f"ee_target_sphere_{arm}").id
        as_["ghost_data"]     = mujoco.MjData(model)
        as_["ghost_data"].qpos[:] = data.qpos[:]

    # Base joint qpos/dof addresses
    _base_x_jnt_adr   = model.jnt_qposadr[model.joint("base_x").id]
    _base_y_jnt_adr   = model.jnt_qposadr[model.joint("base_y").id]
    _base_yaw_adr     = model.jnt_qposadr[model.joint("base_yaw").id]
    _base_x_dof_adr   = model.jnt_dofadr[model.joint("base_x").id]
    _base_y_dof_adr   = model.jnt_dofadr[model.joint("base_y").id]
    _base_yaw_dof_adr = model.jnt_dofadr[model.joint("base_yaw").id]

    # Swerve steering joint qpos/dof addresses — needed for flip and kinematic steering
    _steer_jnt_adr = {
        name: model.jnt_qposadr[model.joint(f"{name}_wheel_steer_joint").id]
        for name, *_ in SWERVE_MODULES
    }
    _steer_dof_adr = {
        name: model.jnt_dofadr[model.joint(f"{name}_wheel_steer_joint").id]
        for name, *_ in SWERVE_MODULES
    }
    _drive_dof_adr = {
        name: model.jnt_dofadr[model.joint(f"{name}_wheel_drive_joint").id]
        for name, *_ in SWERVE_MODULES
    }

    # Lift state — direct joint control, no CuRobo
    _lift_body_id     = model.body("lift_target").id
    _lift_joint_id    = model.joint(LIFT_JOINT).id
    _lift_jnt_adr     = model.jnt_qposadr[_lift_joint_id]
    lift_state = {
        "mocap_id":       model.body_mocapid[_lift_body_id],
        "sphere_geom_id": model.geom("lift_target_sphere").id,
        "ghost_data":     mujoco.MjData(model),
        "tgt_base":       _lift_home_base.copy(),
        "prev_tgt_base":  _lift_home_base.copy(),
        "last_set_world": None,
        "target_val":     0.0,    # desired lift_joint value
        "ghost_ok":       False,  # True when sphere has moved from home
        "state":          IDLE,
        "exec_from":      0.0,    # lift value at start of execution
        "exec_to":        0.0,
        "exec_start":     0.0,
    }
    lift_state["ghost_data"].qpos[:] = data.qpos[:]

    # Explicitly put every arm at its home position on startup so the sim
    # always opens with the robot at home regardless of XML defaults or any
    # state left over from a previous run.
    for arm in arms:
        for name, val in ARM_CONFIGS[arm]["home_joints"].items():
            if name in ctrl_idx:
                data.ctrl[ctrl_idx[name]] = val
            try:
                data.qpos[model.jnt_qposadr[model.joint(name).id]] = val
            except Exception:
                pass
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    # Snap the target spheres to the actual home EE positions after FK
    for arm, as_ in arm_states.items():
        mid = as_["mocap_id"]
        data.mocap_pos[mid]       = get_ee_pos(model, data, as_["cfg"]["ee_site"])
        data.mocap_quat[mid]      = [1.0, 0.0, 0.0, 0.0]
        as_["last_planned_pos"]   = data.mocap_pos[mid].copy()
        as_["prev_sphere_pos"]    = data.mocap_pos[mid].copy()
    # Snap lift sphere to actual arm_base_link world position
    _arm_base_body_id = model.body("arm_base_link").id
    data.mocap_pos[lift_state["mocap_id"]] = data.xpos[_arm_base_body_id].copy()

    # Reference frames at startup
    base_body_id   = model.body("base_link").id
    _bmat0 = data.xmat[base_body_id].reshape(3, 3).copy()
    _bpos0 = data.xpos[base_body_id].copy()
    # Arm EE spheres track arm_base_link (moves with both base driving AND lift)
    _arm_base_pos0 = data.xpos[_arm_base_body_id].copy()  # initial arm_base_link world pos
    for arm, as_ in arm_states.items():
        w = data.mocap_pos[as_["mocap_id"]].copy()
        # tgt_base is now in arm_base_link frame — follows lift changes automatically
        as_["tgt_base"]          = w - _arm_base_pos0   # identity rotation at startup
        as_["prev_tgt_base"]     = w - _arm_base_pos0
        as_["last_planned_base"] = w - _arm_base_pos0
        as_["last_set_world"]    = w.copy()
    # Lift sphere stays relative to base_link only (it IS the target for arm_base_link)
    lw = data.mocap_pos[lift_state["mocap_id"]].copy()
    lift_state["tgt_base"]       = _bmat0.T @ (lw - _bpos0)
    lift_state["prev_tgt_base"]  = _bmat0.T @ (lw - _bpos0)
    lift_state["last_set_world"] = lw.copy()

    # Active arm for keyboard arrow-key control (T to cycle between arms)
    all_targets = arms   # lift has dedicated +/- keys, no cycling needed
    active_arm = [arms[0]]

    # Steer angle commands tracked per module for rate limiting
    _steer_cmd        = {name: 0.0            for name, *_ in SWERVE_MODULES}
    _reversal_phase   = {name: REVERSAL_NORMAL for name, *_ in SWERVE_MODULES}
    _wheel_spd_scale  = {name: 1.0            for name, *_ in SWERVE_MODULES}
    _prev_wheel_dir   = {name: 1.0            for name, *_ in SWERVE_MODULES}

    # Shared events and mutable state
    step_size    = [0.02]
    move         = np.zeros(3)
    lift_move    = [0.0]       # delta for +/- lift keys (always active)
    _glfw_window = [None]        # GLFW window handle, captured on first key event
    _glfw_intercept_installed = [False]  # True once GLFW key callback is replaced
    _opt_flags0 = [None]        # snapshot of viewer.opt.flags; restored every frame
    _scn_flags0 = [None]        # snapshot of viewer.user_scn.flags; restored every frame
    _cmd_vel_pub      = [None]  # rclpy publisher for /cmd_vel (robot mode only)
    _RosTwist         = [None]  # geometry_msgs.msg.Twist class
    _last_cmd_vel_t   = [0.0]   # last publish timestamp for 20 Hz throttle
    _ros_drive_node_holder = [None]  # node reference kept for spin thread
    # Real robot base pose injected from /swerve_drive_controller/odom (robot mode only)
    _real_base = {"x": 0.0, "y": 0.0, "yaw": 0.0,
                  "vx_world": 0.0, "vy_world": 0.0, "wz": 0.0,
                  "received": False}
    _real_base_lock = threading.Lock()
    move_lock    = threading.Lock()
    plan_flag    = threading.Event()
    replay_flag  = threading.Event()
    reset_flag   = threading.Event()
    orient_lock  = threading.Lock()
    orient_delta = np.zeros(3)
    axis_lock    = [None]

    def _axis_label(a):
        return {'x': 'X (red)', 'y': 'Y (green)', 'z': 'Z (blue)', None: 'free'}[a]

    def key_callback(keycode):
        if _glfw_window[0] is None:
            _glfw_window[0] = glfw.get_current_context()
        with move_lock:
            s = step_size[0]
            az_rad    = np.deg2rad(viewer.cam.azimuth)
            cam_right = np.array([-np.sin(az_rad),  np.cos(az_rad), 0.0])
            cam_fwd   = np.array([-np.cos(az_rad), -np.sin(az_rad), 0.0])
            if   keycode == KEY_RIGHT:    move[:] += s * cam_right
            elif keycode == KEY_LEFT:     move[:] -= s * cam_right
            elif keycode == KEY_UP:       move[:] += s * cam_fwd
            elif keycode == KEY_DOWN:     move[:] -= s * cam_fwd
            elif keycode == KEY_PAGE_UP:  move[2] += s
            elif keycode == KEY_PAGE_DN:  move[2] -= s
            elif keycode == KEY_LBRACKET:
                step_size[0] = max(0.005, step_size[0] / 2)
                print(f"[target] step = {step_size[0]*100:.1f} cm")
            elif keycode == KEY_RBRACKET:
                step_size[0] = min(0.20, step_size[0] * 2)
                print(f"[target] step = {step_size[0]*100:.1f} cm")
            elif keycode == KEY_SPACE:
                plan_flag.set()
            elif keycode == KEY_R:        replay_flag.set()
            elif keycode == KEY_H:        reset_flag.set()
            elif keycode == KEY_T and len(arms) > 1:
                idx = arms.index(active_arm[0])
                active_arm[0] = arms[(idx + 1) % len(arms)]
                print(f"[target] Active arm: {active_arm[0]}")
            elif keycode == KEY_LIFT_UP:
                lift_move[0] += step_size[0]
            elif keycode == KEY_LIFT_DOWN:
                lift_move[0] -= step_size[0]
            elif keycode == KEY_ESC:
                axis_lock[0] = None
                print("[drag] Axis constraint: free")
            elif keycode == KEY_X:
                axis_lock[0] = None if axis_lock[0] == 'x' else 'x'
                print(f"[drag] Axis constraint: {_axis_label(axis_lock[0])}")
            elif keycode == KEY_Y:
                axis_lock[0] = None if axis_lock[0] == 'y' else 'y'
                print(f"[drag] Axis constraint: {_axis_label(axis_lock[0])}")
            elif keycode == KEY_ZKEY:
                axis_lock[0] = None if axis_lock[0] == 'z' else 'z'
                print(f"[drag] Axis constraint: {_axis_label(axis_lock[0])}")
        if args.orientation:
            with orient_lock:
                if   keycode == KEY_I: orient_delta[0] += ROT_STEP   # pitch +Y
                elif keycode == KEY_K: orient_delta[0] -= ROT_STEP
                elif keycode == KEY_J: orient_delta[1] += ROT_STEP   # yaw +Z
                elif keycode == KEY_L: orient_delta[1] -= ROT_STEP
                elif keycode == KEY_U: orient_delta[2] += ROT_STEP   # roll +X
                elif keycode == KEY_O: orient_delta[2] -= ROT_STEP

    if args.robot:
        try:
            import rclpy
            from geometry_msgs.msg import Twist as _TwistCls
            from nav_msgs.msg import Odometry as _OdomCls
            import math as _math
            if not rclpy.ok():
                rclpy.init()
            _ros_drive_node = rclpy.create_node("ffw_drive_pub")
            _cmd_vel_pub[0] = _ros_drive_node.create_publisher(_TwistCls, "/cmd_vel", 10)
            _RosTwist[0] = _TwistCls

            def _odom_cb(msg):
                q = msg.pose.pose.orientation
                # quaternion → yaw
                yaw = _math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                # twist is in body frame → rotate to world frame
                vxb = msg.twist.twist.linear.x
                vyb = msg.twist.twist.linear.y
                with _real_base_lock:
                    _real_base["x"]        = msg.pose.pose.position.x
                    _real_base["y"]        = msg.pose.pose.position.y
                    _real_base["yaw"]      = yaw
                    _real_base["vx_world"] = _math.cos(yaw) * vxb - _math.sin(yaw) * vyb
                    _real_base["vy_world"] = _math.sin(yaw) * vxb + _math.cos(yaw) * vyb
                    _real_base["wz"]       = msg.twist.twist.angular.z
                    _real_base["received"] = True

            _ros_drive_node.create_subscription(
                _OdomCls, "/swerve_drive_controller/odom", _odom_cb, 10)
            _ros_drive_node_holder[0] = _ros_drive_node

            _spin_thread = threading.Thread(
                target=rclpy.spin, args=(_ros_drive_node,), daemon=True)
            _spin_thread.start()
            print("[ros2] /cmd_vel publisher + odom subscriber ready")
        except ImportError:
            print("[ros2] rclpy not found — drive will not be sent to real robot")
        except Exception as e:
            print(f"[ros2] Drive publisher error: {e}")

    with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback,
            show_left_ui=False) as viewer:

        viewer.cam.distance  = 3.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 135

        _opt_flags0[0] = viewer.opt.flags.copy()          # baseline — restored every frame
        _scn_flags0[0] = viewer.user_scn.flags.copy()    # baseline — restored every frame
        mujoco_dt = model.opt.timestep

        try:
          while viewer.is_running():
            step_start = time.perf_counter()

            # ── Replace GLFW key callback once window handle is available ──
            # Done from the main loop (not from within key_callback) to avoid
            # GLFW re-entrancy. Replaces MuJoCo's C++ handler so W/S/D no
            # longer toggle mjRND_WIREFRAME / mjRND_SHADOW / mjVIS_STATIC.
            # show_left_ui=False at launch prevents any shortcut on first press;
            # re-enable the panel here once the intercept is in place.
            if not _glfw_intercept_installed[0] and _glfw_window[0] is not None:
                def _key_intercept(window, key, scancode, action, mods):
                    if action == glfw.PRESS or action == glfw.REPEAT:
                        key_callback(key)
                glfw.set_key_callback(_glfw_window[0], _key_intercept)
                _glfw_intercept_installed[0] = True
                try:
                    viewer._sim().ui0_enable = True
                except Exception:
                    pass

            # ── Target spheres follow robot base (drift-free) ─────────────
            cur_base_pos     = data.xpos[base_body_id].copy()
            cur_base_mat     = data.xmat[base_body_id].reshape(3, 3).copy()
            # Arm EE spheres track arm_base_link so they follow lift changes too
            cur_arm_base_pos = data.xpos[_arm_base_body_id].copy()
            cur_arm_base_mat = data.xmat[_arm_base_body_id].reshape(3, 3).copy()
            for arm, as_ in arm_states.items():
                mid = as_["mocap_id"]
                actual = data.mocap_pos[mid].copy()
                # If mocap was modified externally (mouse drag / axis lock), pull into arm_base frame
                if np.linalg.norm(actual - as_["last_set_world"]) > 1e-6:
                    as_["tgt_base"] = cur_arm_base_mat.T @ (actual - cur_arm_base_pos)
                # Recompute world position fresh from stable arm_base-frame value — no drift
                new_world = cur_arm_base_mat @ as_["tgt_base"] + cur_arm_base_pos
                data.mocap_pos[mid]    = new_world
                as_["last_set_world"]  = new_world.copy()
                as_["prev_sphere_pos"] = new_world.copy()
                # last_planned_pos tracks arm_base movement so plan_changed reflects
                # sphere displacement relative to robot, not world displacement
                as_["last_planned_pos"] = cur_arm_base_mat @ as_["last_planned_base"] + cur_arm_base_pos
            # Lift sphere: detect drag (only Z in base frame matters),
            # clamp to joint range, keep XY fixed at arm_base_link offset
            l_mid    = lift_state["mocap_id"]
            l_actual = data.mocap_pos[l_mid].copy()
            if np.linalg.norm(l_actual - lift_state["last_set_world"]) > 1e-6:
                new_tgt = cur_base_mat.T @ (l_actual - cur_base_pos)
                # Only Z changes; clamp Z to lift range
                new_z   = np.clip(new_tgt[2] - LIFT_ARM_BASE_OFFSET[2],
                                  LIFT_RANGE[0], LIFT_RANGE[1])
                lift_state["tgt_base"] = np.array([LIFT_ARM_BASE_OFFSET[0],
                                                    LIFT_ARM_BASE_OFFSET[1],
                                                    LIFT_ARM_BASE_OFFSET[2] + new_z])
                lift_state["target_val"] = new_z
            l_new_world = cur_base_mat @ lift_state["tgt_base"] + cur_base_pos
            data.mocap_pos[l_mid]          = l_new_world
            lift_state["last_set_world"]   = l_new_world.copy()

            # ── H — reset all arms to home ─────────────────────────────────
            if reset_flag.is_set():
                reset_flag.clear()
                plan_flag.clear()
                replay_flag.clear()
                for arm, as_ in arm_states.items():
                    as_["state"]            = IDLE
                    as_["ghost_traj"]       = None
                    as_["ghost_joint_names"] = None
                    as_["ghost_ok"]         = False
                    as_["last_move_time"]   = 0.0
                    for name, val in as_["cfg"]["home_joints"].items():
                        if name in ctrl_idx:
                            data.ctrl[ctrl_idx[name]] = val
                        try:
                            data.qpos[model.jnt_qposadr[model.joint(name).id]] = val
                        except Exception:
                            pass
                # Reset lift joint alongside arms, then run one FK pass
                data.ctrl[ctrl_idx[LIFT_JOINT]] = 0.0
                data.qpos[_lift_jnt_adr] = 0.0
                data.qvel[:] = 0
                data.qacc[:] = 0
                mujoco.mj_forward(model, data)
                # Read arm_base_link pose after full-home FK
                _rst_arm_base_pos = data.xpos[_arm_base_body_id].copy()
                _rst_arm_base_mat = data.xmat[_arm_base_body_id].reshape(3, 3).copy()
                for arm, as_ in arm_states.items():
                    mid = as_["mocap_id"]
                    data.mocap_pos[mid]  = get_ee_pos(model, data, as_["cfg"]["ee_site"])
                    data.mocap_quat[mid] = [1.0, 0.0, 0.0, 0.0]
                    w = data.mocap_pos[mid].copy()
                    as_["last_planned_pos"]   = w.copy()
                    as_["last_planned_base"]  = _rst_arm_base_mat.T @ (w - _rst_arm_base_pos)
                    as_["last_planned_quat"]  = data.mocap_quat[mid].copy()
                    as_["tgt_base"]           = _rst_arm_base_mat.T @ (w - _rst_arm_base_pos)
                    as_["prev_tgt_base"]      = _rst_arm_base_mat.T @ (w - _rst_arm_base_pos)
                    as_["last_set_world"]     = w.copy()
                    model.geom_rgba[as_["sphere_geom_id"]] = RGBA_IDLE
                # Snap lift sphere to home arm_base_link position
                lw = data.xpos[_arm_base_body_id].copy()
                data.mocap_pos[lift_state["mocap_id"]] = lw
                lift_state["tgt_base"]       = cur_base_mat.T @ (lw - cur_base_pos)
                lift_state["prev_tgt_base"]  = cur_base_mat.T @ (lw - cur_base_pos)
                lift_state["last_set_world"] = lw.copy()
                lift_state["target_val"]     = 0.0
                lift_state["ghost_ok"]       = False
                lift_state["state"]          = IDLE
                model.geom_rgba[lift_state["sphere_geom_id"]] = RGBA_IDLE
                print("[home] Reset. Arms and lift back at home.")

            # ── Keyboard position moves ────────────────────────────────────
            # Poll GLFW key state so drive is active only while keys are held
            _win = _glfw_window[0]
            if _win is not None:
                _vx = 0.0
                _wz = 0.0
                if glfw.get_key(_win, KEY_W) == glfw.PRESS: _vx += DRIVE_LIN_VEL
                if glfw.get_key(_win, KEY_S) == glfw.PRESS: _vx -= DRIVE_LIN_VEL
                if glfw.get_key(_win, KEY_A) == glfw.PRESS: _wz += DRIVE_ANG_VEL
                if glfw.get_key(_win, KEY_D) == glfw.PRESS: _wz -= DRIVE_ANG_VEL
            else:
                _vx = 0.0
                _wz = 0.0
            # Forward drive command to real robot at 20 Hz (--robot mode)
            if _cmd_vel_pub[0] is not None:
                _now_cv = time.perf_counter()
                if _now_cv - _last_cmd_vel_t[0] >= 0.05:
                    _tw = _RosTwist[0]()
                    _tw.linear.x  = float(_vx)
                    _tw.angular.z = float(_wz)
                    _cmd_vel_pub[0].publish(_tw)
                    _last_cmd_vel_t[0] = _now_cv
            with move_lock:
                # +/- keys: always control lift regardless of active target
                if lift_move[0] != 0:
                    new_val = np.clip(lift_state["target_val"] + lift_move[0],
                                      LIFT_RANGE[0], LIFT_RANGE[1])
                    lift_state["target_val"] = new_val
                    lift_state["tgt_base"] = np.array([
                        LIFT_ARM_BASE_OFFSET[0],
                        LIFT_ARM_BASE_OFFSET[1],
                        LIFT_ARM_BASE_OFFSET[2] + new_val])
                    lw = cur_base_mat @ lift_state["tgt_base"] + cur_base_pos
                    data.mocap_pos[lift_state["mocap_id"]] = lw
                    lift_state["last_set_world"] = lw.copy()
                    print(f"[lift] target={new_val:+.3f} m")
                    lift_move[0] = 0.0
                # Arrow/Page keys: control active arm
                if np.any(move != 0):
                    aa  = arm_states[active_arm[0]]
                    mid = aa["mocap_id"]
                    data.mocap_pos[mid] += move
                    pos = data.mocap_pos[mid]
                    aa["tgt_base"]       = cur_arm_base_mat.T @ (pos - cur_arm_base_pos)
                    aa["last_set_world"] = pos.copy()
                    print(f"[target] [{active_arm[0]}] "
                          f"{pos[0]:.3f}  {pos[1]:.3f}  {pos[2]:.3f}")
                    move[:] = 0

            # ── Swerve IK (matches ffw_swerve_drive_controller.cpp) ──────────
            _steer_max_delta = STEER_RATE * mujoco_dt
            _cmd_zero = (_vx == 0.0 and _wz == 0.0)
            _all_aligned  = True   # synchronized gating: ALL modules must align
            _pending_drive = {}    # mod → final wheel vel before gating/saturation

            for _mod, _mx, _my, _sa, _da, _aoff in SWERVE_MODULES:
                if _cmd_zero:
                    data.ctrl[ctrl_idx[_da]]        = 0.0
                    data.qvel[_drive_dof_adr[_mod]] = 0.0
                    _reversal_phase[_mod]  = REVERSAL_NORMAL
                    _wheel_spd_scale[_mod] = 1.0
                    continue  # leave steer at current commanded position

                # IK — apply per-module angle offset (module_angle_offsets in config)
                _vwx = _vx - _wz * _my
                _vwy =       _wz * _mx
                _raw_steer = np.arctan2(_vwy, _vwx + 1e-9) - _aoff
                _wspd = np.hypot(_vwx, _vwy) / WHEEL_RADIUS

                # Read actual joint position — used for flip (matches real controller)
                _actual_steer = data.qpos[_steer_jnt_adr[_mod]]

                # 180° flip vs ACTUAL joint angle (swerve_drive_controller.cpp §4.3)
                _fdiff = (_raw_steer - _actual_steer + np.pi) % (2 * np.pi) - np.pi
                if abs(_fdiff) > np.pi / 2:
                    _raw_steer = (_raw_steer + np.pi + np.pi) % (2 * np.pi) - np.pi
                    _wspd = -_wspd

                # Boundary crossing double-flip (§4.3.1)
                _diff_after = (_raw_steer - _actual_steer + np.pi) % (2 * np.pi) - np.pi
                _crosses = ((_actual_steer > 0 and _raw_steer < 0 and _diff_after > 0) or
                            (_actual_steer < 0 and _raw_steer > 0 and _diff_after < 0))
                if _crosses:
                    _raw_steer = (_raw_steer + np.pi + np.pi) % (2 * np.pi) - np.pi
                    _wspd *= -1.0

                # Wheel direction for reversal FSM
                _wdir = 1.0 if _wspd >= 0.0 else -1.0
                _cur_wheel_vel = abs(data.qvel[_drive_dof_adr[_mod]])

                # Start reversal on any direction change (matches real controller — no minimum speed guard)
                if (_wdir != _prev_wheel_dir[_mod]
                        and _reversal_phase[_mod] == REVERSAL_NORMAL):
                    _reversal_phase[_mod] = REVERSAL_DECEL

                # In DECEL: hold steer at current joint angle (don't change orientation)
                _steer_tgt = _actual_steer if _reversal_phase[_mod] == REVERSAL_DECEL else _raw_steer

                # Kinematic steering: rate-limit and set joint directly (mirrors real motor
                # which tracks position at 100 rad/s — no force/friction fight needed).
                _prev_cmd = _steer_cmd[_mod]
                _sdiff = (_steer_tgt - _prev_cmd + np.pi) % (2 * np.pi) - np.pi
                _sdiff = np.clip(_sdiff, -_steer_max_delta, _steer_max_delta)
                _new_cmd = _prev_cmd + _sdiff
                _steer_cmd[_mod] = _new_cmd
                data.qpos[_steer_jnt_adr[_mod]] = _new_cmd
                data.qvel[_steer_dof_adr[_mod]] = 0.0  # zero vel so integrator doesn't drift past target
                data.ctrl[ctrl_idx[_sa]] = _new_cmd  # keep actuator at target so it resists perturbations

                # Alignment error (always vs raw IK target, not DECEL-held target)
                _aerr = abs((_raw_steer - _actual_steer + np.pi) % (2 * np.pi) - np.pi)
                # 2-tier threshold: strict when stationary, loose when spinning
                _athresh = STEER_ALIGN_THRESH if _cur_wheel_vel >= 0.1 else STEER_ALIGN_THRESH_IDLE
                if _aerr > _athresh:
                    _all_aligned = False

                # Reversal FSM transitions
                if _reversal_phase[_mod] == REVERSAL_DECEL:
                    _wheel_spd_scale[_mod] = max(0.0,
                        _wheel_spd_scale[_mod] - REVERSAL_DECEL_RATE * mujoco_dt)
                    if _wheel_spd_scale[_mod] <= REVERSAL_THRESHOLD:
                        _wheel_spd_scale[_mod] = 0.0
                        _reversal_phase[_mod] = REVERSAL_STEERING
                elif _reversal_phase[_mod] == REVERSAL_STEERING:
                    if _aerr < REVERSAL_STEER_TOL:
                        _prev_wheel_dir[_mod] = _wdir
                        _reversal_phase[_mod] = REVERSAL_ACCEL
                elif _reversal_phase[_mod] == REVERSAL_ACCEL:
                    _wheel_spd_scale[_mod] = min(1.0,
                        _wheel_spd_scale[_mod] + REVERSAL_ACCEL_RATE * mujoco_dt)
                    if _wheel_spd_scale[_mod] >= 1.0:
                        _reversal_phase[_mod] = REVERSAL_NORMAL
                else:  # NORMAL
                    _wheel_spd_scale[_mod] = 1.0
                    _prev_wheel_dir[_mod] = _wdir

                # During DECEL keep old direction; otherwise use new direction
                _eff_dir = _prev_wheel_dir[_mod] if _reversal_phase[_mod] == REVERSAL_DECEL else _wdir
                _pending_drive[_mod] = _eff_dir * abs(_wspd) * _wheel_spd_scale[_mod]

            # Second pass: synchronized gating + saturation scaling
            if not _cmd_zero:
                _sat = 1.0
                for _v in _pending_drive.values():
                    if abs(_v) > WHEEL_SPEED_LIMIT:
                        _sat = min(_sat, WHEEL_SPEED_LIMIT / abs(_v))
                for _mod, _mx, _my, _sa, _da, _aoff in SWERVE_MODULES:
                    if _mod in _pending_drive:
                        _drv = _pending_drive[_mod] * _sat if _all_aligned else 0.0
                        data.ctrl[ctrl_idx[_da]]    = _drv
                        data.qvel[_drive_dof_adr[_mod]] = _drv  # kinematic: bypass contact solver

            # ── Base motion ────────────────────────────────────────────────
            if args.robot:
                # Robot mode: inject real odometry so sim mirrors real base exactly.
                # Base actuators are zeroed — the physics joint is driven kinematically.
                with _real_base_lock:
                    if _real_base["received"]:
                        data.qpos[_base_x_jnt_adr]   = _real_base["x"]
                        data.qpos[_base_y_jnt_adr]    = _real_base["y"]
                        data.qpos[_base_yaw_adr]      = _real_base["yaw"]
                        data.qvel[_base_x_dof_adr]    = _real_base["vx_world"]
                        data.qvel[_base_y_dof_adr]    = _real_base["vy_world"]
                        data.qvel[_base_yaw_dof_adr]  = _real_base["wz"]
                data.ctrl[ctrl_idx["base_x_vel"]]   = 0.0
                data.ctrl[ctrl_idx["base_y_vel"]]   = 0.0
                data.ctrl[ctrl_idx["base_yaw_vel"]] = 0.0
            else:
                # Sim-only mode: direct joint velocity drive (bypasses wheel-floor friction).
                # base_x/y are world-frame axes, so rotate body-frame _vx by current yaw.
                _yaw = data.qpos[_base_yaw_adr]
                if _cmd_zero or not _all_aligned:
                    data.ctrl[ctrl_idx["base_x_vel"]]   = 0.0
                    data.ctrl[ctrl_idx["base_y_vel"]]   = 0.0
                    data.ctrl[ctrl_idx["base_yaw_vel"]] = 0.0
                else:
                    data.ctrl[ctrl_idx["base_x_vel"]]   = _vx * np.cos(_yaw)
                    data.ctrl[ctrl_idx["base_y_vel"]]   = _vx * np.sin(_yaw)
                    data.ctrl[ctrl_idx["base_yaw_vel"]] = _wz

            # ── Keyboard orientation (active arm) ─────────────────────────
            if args.orientation:
                mid = arm_states[active_arm[0]]["mocap_id"]
                with orient_lock:
                    delta = orient_delta.copy()
                    orient_delta[:] = 0
                if np.any(delta != 0):
                    q = data.mocap_quat[mid].copy()
                    if delta[0] != 0:
                        q = _rotate_world(q, np.array([0.0, 1.0, 0.0]), delta[0])
                    if delta[1] != 0:
                        q = _rotate_world(q, np.array([0.0, 0.0, 1.0]), delta[1])
                    if delta[2] != 0:
                        q = _rotate_world(q, np.array([1.0, 0.0, 0.0]), delta[2])
                    data.mocap_quat[mid] = q
                    print(f"[target] [{active_arm[0]}] quat "
                          f"{q[0]:.3f}  {q[1]:.3f}  {q[2]:.3f}  {q[3]:.3f}")

            # ── Per-arm planning / execution loop ─────────────────────────
            # Lazily load frame offsets if server_info wasn't ready at startup
            for arm in list(_server_info_pending):
                as_ = arm_states[arm]
                try:
                    info = json.loads(as_["server_info"].read_text())
                    if info.get("arm") == arm:
                        curobo_home = np.array(info["ee_home_curobo"])
                        as_["frame_offset"] = init_ee[arm] - curobo_home
                        print(f"[info] [{arm}] Frame offset (MuJoCo−CuRobo): "
                              f"{as_['frame_offset'][0]:+.4f}  "
                              f"{as_['frame_offset'][1]:+.4f}  "
                              f"{as_['frame_offset'][2]:+.4f}")
                        _server_info_pending.discard(arm)
                except Exception:
                    pass

            do_execute = plan_flag.is_set()
            do_replay  = replay_flag.is_set()
            if do_execute:
                plan_flag.clear()
            if do_replay:
                replay_flag.clear()

            for arm, as_ in arm_states.items():
                arm_mid  = as_["mocap_id"]
                cfg      = as_["cfg"]
                EE_SITE  = cfg["ee_site"]
                PLAN_JOINTS = cfg["plan_joints"]
                sgid        = as_["sphere_geom_id"]

                current_pos  = data.mocap_pos[arm_mid].copy()
                current_quat = data.mocap_quat[arm_mid].copy()
                # Use base-frame delta to detect user-initiated moves (keyboard or mouse drag).
                # World-frame delta is useless here because the base-follow block pre-updates
                # prev_sphere_pos to the current world position every frame.
                base_pos_delta = np.linalg.norm(as_["tgt_base"] - as_["prev_tgt_base"])
                quat_delta     = 1.0 - abs(float(np.dot(current_quat, as_["prev_sphere_quat"])))
                frame_moved    = base_pos_delta > 0.001 or (args.orientation and quat_delta > 0.0005)

                plan_pos_delta  = np.linalg.norm(current_pos  - as_["last_planned_pos"])
                plan_quat_delta = 1.0 - abs(float(np.dot(current_quat, as_["last_planned_quat"])))
                plan_changed    = (plan_pos_delta > 0.003
                                   or (args.orientation and plan_quat_delta > 0.001))

                if frame_moved:
                    as_["last_move_time"] = time.perf_counter()
                    if as_["state"] == WAITING:
                        as_["state"] = IDLE
                        model.geom_rgba[sgid] = RGBA_IDLE  # cancel in-flight plan visually
                    if plan_changed:
                        as_["ghost_traj"]        = None
                        as_["ghost_joint_names"] = None
                        as_["ghost_ok"]          = False
                        model.geom_rgba[sgid] = RGBA_IDLE

                as_["prev_sphere_pos"]  = current_pos.copy()
                as_["prev_sphere_quat"] = current_quat.copy()
                as_["prev_tgt_base"]    = as_["tgt_base"].copy()

                # Auto-plan after debounce
                if (as_["state"] == IDLE
                        and as_["last_move_time"] > 0.0
                        and plan_changed
                        and time.perf_counter() - as_["last_move_time"] > PLAN_DEBOUNCE):
                    _trigger_plan(data, model, current_pos, current_quat,
                                  as_["req_id"] + 1, PLAN_JOINTS, args.orientation,
                                  as_["frame_offset"], as_["goal_file"],
                                  cur_arm_base_pos, cur_arm_base_mat, _arm_base_pos0)
                    as_["req_id"]            += 1
                    as_["last_planned_pos"]   = current_pos.copy()
                    as_["last_planned_base"]  = cur_arm_base_mat.T @ (current_pos - cur_arm_base_pos)
                    as_["last_planned_quat"]  = current_quat.copy()
                    as_["state"]              = WAITING
                    as_["wait_start"]         = time.perf_counter()
                    model.geom_rgba[sgid] = RGBA_PLANNING

                # SPACE — execute all arms with ready plans; re-plan active arm if none
                if do_execute:
                    if as_["traj"] is not None and as_["ghost_ok"] and not plan_changed:
                        T = as_["traj"]["positions"].shape[0]
                        print(f"[plan] [{arm}] Executing {T} waypoints "
                              f"({T * as_['traj']['dt']:.2f}s)")
                        if args.robot:
                            atopic = (args.topic if len(arms) == 1 and args.topic
                                      else cfg["action_topic"])
                            ready_evt = threading.Event()
                            threading.Thread(
                                target=publish_ros2,
                                args=(as_["traj"]["joint_names"],
                                      as_["traj"]["positions"],
                                      as_["traj"]["dt"],
                                      atopic, cfg["gripper_joint"], ready_evt),
                                daemon=True,
                            ).start()
                            print("[exec] Waiting for hardware acceptance...")
                            if not ready_evt.wait(timeout=2.0):
                                print("[exec] No response within 2s — running sim only")
                        as_["traj_start"] = data.time
                        as_["state"]      = EXECUTING
                    elif (arm == active_arm[0]
                          and plan_changed
                          and as_["state"] not in (EXECUTING, WAITING)):
                        _trigger_plan(data, model, current_pos, current_quat,
                                      as_["req_id"] + 1, PLAN_JOINTS, args.orientation,
                                      as_["frame_offset"], as_["goal_file"],
                                      cur_arm_base_pos, cur_arm_base_mat, _arm_base_pos0)
                        as_["req_id"]            += 1
                        as_["last_planned_pos"]   = current_pos.copy()
                        as_["last_planned_base"]  = cur_arm_base_mat.T @ (current_pos - cur_arm_base_pos)
                        as_["last_planned_quat"]  = current_quat.copy()
                        as_["state"]             = WAITING
                        as_["wait_start"]        = time.perf_counter()
                        model.geom_rgba[sgid] = RGBA_PLANNING
                        print(f"[plan] [{arm}] Planning to "
                              f"[{current_pos[0]:.3f}, {current_pos[1]:.3f},"
                              f" {current_pos[2]:.3f}]")

                # R — replay active arm's last trajectory
                if do_replay and arm == active_arm[0]:
                    if as_["traj"] is not None and as_["state"] == IDLE:
                        as_["traj_start"] = data.time
                        as_["state"]      = EXECUTING
                        print(f"[plan] [{arm}] Replaying "
                              f"{as_['traj']['positions'].shape[0]} waypoints")

                # Poll plan result — timeout is checked first, independent of file state
                if as_["state"] == WAITING:
                    if time.perf_counter() - as_["wait_start"] > PLAN_TIMEOUT:
                        as_["state"] = IDLE
                        model.geom_rgba[sgid] = RGBA_IDLE
                        print(f"[plan] [{arm}] Timeout — is curobo_server.py running?")
                    else:
                        goal = read_goal(as_["goal_file"])
                        if goal is not None and goal.get("id") == as_["req_id"]:
                            if goal.get("status") == "done":
                                npz = np.load(as_["traj_file"], allow_pickle=True)
                                as_["traj"] = {
                                    "joint_names": list(npz["joint_names"]),
                                    "positions":   npz["positions"],
                                    "dt":          float(npz["dt"]),
                                }
                                as_["ghost_traj"]        = as_["traj"]["positions"]
                                as_["ghost_traj_dt"]     = as_["traj"]["dt"]
                                as_["ghost_anim_t0"]     = time.perf_counter()
                                as_["ghost_joint_names"] = as_["traj"]["joint_names"]
                                as_["ghost_ok"]          = True
                                model.geom_rgba[sgid] = RGBA_OK
                                as_["state"] = IDLE
                                T = as_["traj"]["positions"].shape[0]
                                # FK accuracy check: how close is planned EE to target?
                                _chk = mujoco.MjData(model)
                                _chk.qpos[:] = data.qpos[:]
                                for _j, _n in enumerate(as_["traj"]["joint_names"]):
                                    try:
                                        _chk.qpos[model.jnt_qposadr[model.joint(_n).id]] = \
                                            float(as_["traj"]["positions"][-1, _j])
                                    except Exception:
                                        pass
                                mujoco.mj_forward(model, _chk)
                                _ee  = _chk.site_xpos[model.site(EE_SITE).id].copy()
                                _err = np.linalg.norm(_ee - data.mocap_pos[arm_mid]) * 100
                                print(f"[plan] [{arm}] Reachable — {T} waypoints"
                                      f" ({T * as_['traj']['dt']:.2f}s)"
                                      f"  EE accuracy: {_err:.1f} cm  SPACE to execute.")
                            elif goal.get("status") == "failed":
                                as_["ghost_traj"]        = None
                                as_["ghost_joint_names"] = None
                                as_["ghost_ok"]          = False
                                model.geom_rgba[sgid] = RGBA_FAIL
                                as_["state"] = IDLE
                                print(f"[plan] [{arm}] Unreachable — move sphere and try again.")

                # Execute trajectory — ctrl only during playback; qpos+zero vel at completion
                if as_["state"] == EXECUTING and as_["traj"] is not None:
                    T       = as_["traj"]["positions"].shape[0]
                    elapsed = data.time - as_["traj_start"]
                    t       = min(int(elapsed / as_["traj"]["dt"]), T - 1)
                    for j, name in enumerate(as_["traj"]["joint_names"]):
                        val = float(as_["traj"]["positions"][t, j])
                        if name in ctrl_idx:
                            data.ctrl[ctrl_idx[name]] = val
                    if elapsed >= T * as_["traj"]["dt"]:
                        as_["state"]             = IDLE
                        as_["traj"]              = None
                        as_["ghost_traj"]        = None
                        as_["ghost_joint_names"] = None
                        as_["ghost_ok"]          = False
                        model.geom_rgba[sgid]     = RGBA_IDLE
                        print(f"[plan] [{arm}] Done.")

            # ── Lift: ghost preview + SPACE execution ─────────────────────
            _lift_cur = data.qpos[_lift_jnt_adr]
            _lift_tgt = lift_state["target_val"]
            _lift_changed = abs(_lift_tgt - _lift_cur) > 0.002
            # Update ghost_ok whenever target differs from current joint
            lift_state["ghost_ok"] = _lift_changed or lift_state["state"] == EXECUTING

            if do_execute and _lift_changed and lift_state["state"] != EXECUTING:
                lift_state["exec_from"]  = _lift_cur
                lift_state["exec_to"]    = _lift_tgt
                lift_state["exec_start"] = data.time
                lift_state["state"]      = EXECUTING
                model.geom_rgba[lift_state["sphere_geom_id"]] = RGBA_PLANNING
                dur = abs(_lift_tgt - _lift_cur) / LIFT_EXEC_SPEED
                print(f"[lift] Executing: {_lift_cur:+.3f} → {_lift_tgt:+.3f} m  "
                      f"({dur:.1f}s)")

            if lift_state["state"] == EXECUTING:
                dur = abs(lift_state["exec_to"] - lift_state["exec_from"]) / LIFT_EXEC_SPEED
                t   = min((data.time - lift_state["exec_start"]) / max(dur, 1e-6), 1.0)
                cur = lift_state["exec_from"] + t * (lift_state["exec_to"] - lift_state["exec_from"])
                data.ctrl[ctrl_idx[LIFT_JOINT]] = cur
                if t >= 1.0:
                    lift_state["state"] = IDLE
                    model.geom_rgba[lift_state["sphere_geom_id"]] = RGBA_IDLE
                    lift_state["ghost_ok"] = False
                    print(f"[lift] Done.")

            # Show ghost of upper body at target lift height
            if lift_state["ghost_ok"]:
                _gd = lift_state["ghost_data"]
                _gd.qpos[:] = data.qpos[:]
                _gd.qpos[_lift_jnt_adr] = lift_state["target_val"]
                model.geom_rgba[lift_state["sphere_geom_id"]] = RGBA_OK

            # ── Physics step ──────────────────────────────────────────────
            mujoco.mj_step(model, data)

            # ── Ghost rendering (all arms + lift) ─────────────────────────
            with viewer.lock():
                viewer.user_scn.ngeom = 0
            for arm, as_ in arm_states.items():
                if as_["ghost_traj"] is not None and as_["ghost_joint_names"] is not None:
                    T_g       = as_["ghost_traj"].shape[0]
                    total_dur = T_g * as_["ghost_traj_dt"]
                    elapsed_g = (time.perf_counter() - as_["ghost_anim_t0"]) % total_dur
                    anim_idx  = min(int(elapsed_g / as_["ghost_traj_dt"]), T_g - 1)
                    set_ghost(viewer, model, as_["ghost_data"], data,
                              as_["ghost_traj"][anim_idx], as_["ghost_joint_names"],
                              ok=as_["ghost_ok"],
                              ghost_links=tuple(as_["cfg"]["ghost_links"]),
                              clear_scene=False)
            # Lift ghost: static pose of upper body at target lift height
            if lift_state["ghost_ok"]:
                set_ghost(viewer, model, lift_state["ghost_data"], data,
                          ghost_joints=None,
                          joint_names=[],
                          ok=True,
                          ghost_links=tuple(LIFT_GHOST_LINKS),
                          clear_scene=False,
                          pre_fk=True)

            if _opt_flags0[0] is not None:
                viewer.opt.flags[:]      = _opt_flags0[0]
                viewer.user_scn.flags[:] = _scn_flags0[0]

            # ── Axis-lock projection on active arm's mocap target ─────────
            active_mid   = arm_states[active_arm[0]]["mocap_id"]
            pre_sync_pos = data.mocap_pos[active_mid].copy()
            viewer.sync()
            if axis_lock[0] is not None:
                delta = data.mocap_pos[active_mid] - pre_sync_pos
                if np.linalg.norm(delta) > 1e-6:
                    ax = _AXIS_VEC[axis_lock[0]]
                    data.mocap_pos[active_mid] = pre_sync_pos + np.dot(delta, ax) * ax
                    pos = data.mocap_pos[active_mid]
                    print(f"[target] [{active_arm[0]}] "
                          f"{pos[0]:.3f}  {pos[1]:.3f}  {pos[2]:.3f}")

            remaining = mujoco_dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

        except KeyboardInterrupt:
            pass
        finally:
            if _cmd_vel_pub[0] is not None and _RosTwist[0] is not None:
                _cmd_vel_pub[0].publish(_RosTwist[0]())  # zero Twist → stop robot
            os._exit(0)


def _trigger_plan(data, model, target_pos: np.ndarray, target_quat: np.ndarray,
                  req_id: int, plan_joints: list, use_orientation: bool,
                  frame_offset: np.ndarray, goal_file: Path,
                  base_pos: np.ndarray | None = None,
                  base_mat: np.ndarray | None = None,
                  base_pos_init: np.ndarray | None = None):
    joint_state = {}
    for name in plan_joints:
        try:
            joint_state[name] = round(
                data.qpos[model.jnt_qposadr[model.joint(name).id]].item(), 3)
        except Exception:
            joint_state[name] = 0.0
    if base_pos is not None and base_mat is not None:
        # Convert world target to base-relative, then add back the initial base
        # position so that frame_offset (which encodes the FK difference relative
        # to the initial MuJoCo world frame) stays valid.
        target_curobo = (base_mat.T @ (target_pos - base_pos)
                         + (base_pos_init if base_pos_init is not None else np.zeros(3)))
    else:
        target_curobo = target_pos
    goal_curobo = (target_curobo - frame_offset).tolist()
    orientation = target_quat.tolist() if use_orientation else None
    send_goal(goal_curobo, req_id, joint_state, orientation, goal_file)


if __name__ == "__main__":
    main()
