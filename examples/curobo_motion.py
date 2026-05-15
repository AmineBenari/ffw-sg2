#!/usr/bin/env python3
"""
curobo_motion.py — CuRobo motion planning for the FFW-SG2 left arm.

Plans a collision-free trajectory and saves it to robot/traj.npz.
Run INSIDE the Docker container (requires GPU + CuRobo):

    docker compose run ffw-sg2-planner python3 -u examples/curobo_motion.py [OPTIONS]

Goal options (mutually exclusive):
    --goal  X Y Z        Absolute EE target in world frame (metres)
    --delta DX DY DZ     Offset from the home EE position (metres)
    (neither)            Auto-try ±10 cm along each axis, pick first success

Examples:
    python3 -u examples/curobo_motion.py --goal 0.45 0.47 1.05
    python3 -u examples/curobo_motion.py --delta 0 0 0.15
    python3 -u examples/curobo_motion.py

Then visualise / execute:
    python3 examples/play_trajectory.py             # sim only
    python3 examples/play_trajectory.py --robot     # sim + real robot
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import yaml

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState, GoalToolPose

REPO_ROOT = Path(__file__).parent.parent
URDF_SRC  = REPO_ROOT / "robot/urdf/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"
URDF_OUT  = REPO_ROOT / "robot/urdf/ffw_sg2_processed.urdf"
ROBOT_CFG = REPO_ROOT / "robot/curobo/ffw_sg2_left_arm.yml"
TRAJ_OUT  = REPO_ROOT / "robot/traj.npz"


# ── Step 1: Preprocess URDF ───────────────────────────────────────────────

def prepare_urdf() -> Path:
    content  = URDF_SRC.read_text()
    pkg_root = str(REPO_ROOT / "robot/urdf")
    content  = content.replace("package://ffw_description/", pkg_root + "/")
    content  = re.sub(
        r'<mesh filename="file:///[^"]*\.stl"/>',
        '<box size="0.02 0.02 0.02"/>',
        content,
    )
    URDF_OUT.write_text(content)
    print(f"[urdf] written to {URDF_OUT.name}")
    return URDF_OUT


# ── Step 2: Load CuRobo config ────────────────────────────────────────────

def load_robot_config() -> dict:
    cfg = yaml.safe_load(ROBOT_CFG.read_text())
    kin = cfg["robot_cfg"]["kinematics"]
    kin["urdf_path"]       = str(REPO_ROOT / kin["urdf_path"])
    kin["asset_root_path"] = str(REPO_ROOT / kin["asset_root_path"])
    return cfg


# ── Step 3: Plan ──────────────────────────────────────────────────────────

def plan(robot_cfg_dict: dict,
         goal_world: list[float] | None = None,
         delta: list[float] | None = None):
    """
    Plan a collision-free trajectory for the left arm.

    goal_world : [x, y, z] absolute EE target in world frame.
    delta      : [dx, dy, dz] offset from home EE (resolved via FK).
    Neither    : auto-try ±10 cm deltas, accept first success.

    Returns (joint_names, positions, dt) or (None, None, None) on failure.
    positions shape: (T, dof).  dt: seconds between waypoints.
    """
    config  = MotionPlannerCfg.create(robot=robot_cfg_dict)
    planner = MotionPlanner(config)

    print("[curobo] Warming up (compiles CUDA graphs on first call)...")
    planner.warmup(enable_graph=True, num_warmup_iterations=3)

    q_home = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )

    fk_state = planner.kinematics.compute_kinematics(q_home)
    ee_pose  = fk_state.tool_poses.get_link_pose(planner.tool_frames[0])
    ee_pos   = ee_pose.position.squeeze()    # (3,)
    ee_quat  = ee_pose.quaternion.squeeze()  # (4,) wxyz

    print(f"[curobo] Home EE position : {ee_pos.cpu().tolist()}")

    # Build the list of candidate goal positions to try
    if goal_world is not None:
        candidates = [torch.tensor(goal_world, dtype=torch.float32, device=ee_pos.device)]
    elif delta is not None:
        d = torch.tensor(delta, dtype=torch.float32, device=ee_pos.device)
        candidates = [ee_pos.clone() + d]
    else:
        offsets = [(0.10,0,0), (-0.10,0,0), (0,-0.10,0), (0,0,0.10)]
        candidates = [ee_pos.clone() + torch.tensor(o, device=ee_pos.device) for o in offsets]

    result = None
    for goal_pos in candidates:
        print(f"[curobo] Trying {goal_pos.cpu().tolist()}")
        goal = GoalToolPose(
            tool_frames=planner.tool_frames,
            position=goal_pos.reshape(1, 1, 1, 1, 3),
            quaternion=ee_quat.reshape(1, 1, 1, 1, 4),
        )
        result = planner.plan_pose(goal, q_home)
        if result is not None and result.success.any():
            print(f"[curobo] Accepted → {goal_pos.cpu().tolist()}")
            break
        print("[curobo] Rejected, trying next...")

    if result is None or not result.success.any():
        print("[curobo] Planning FAILED.")
        return None, None, None

    traj = result.get_interpolated_plan()
    dt   = planner.trajopt_solver.config.interpolation_dt
    dof  = traj.position.shape[-1]
    pos  = traj.position.reshape(-1, dof)
    T    = pos.shape[0]
    print(f"[curobo] Success: {T} waypoints, {T * dt:.2f}s")

    names = (
        traj.joint_names
        if hasattr(traj, "joint_names") and traj.joint_names is not None
        else planner.joint_names
    )
    return names, pos, dt


# ── Step 4: Save ──────────────────────────────────────────────────────────

def save(joint_names, positions: torch.Tensor, dt: float):
    np.savez(
        TRAJ_OUT,
        joint_names=np.array(joint_names),
        positions=positions.cpu().numpy(),
        dt=np.float64(dt),
    )
    print(f"[traj] saved {positions.shape[0]} waypoints → {TRAJ_OUT}")
    print(f"[traj] play:  python3 examples/play_trajectory.py")
    print(f"[traj] + bot: python3 examples/play_trajectory.py --robot")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CuRobo motion planner for FFW-SG2 left arm")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--goal",  nargs=3, type=float, metavar=("X","Y","Z"),
                       help="Absolute EE target in world frame (m)")
    group.add_argument("--delta", nargs=3, type=float, metavar=("DX","DY","DZ"),
                       help="EE offset from home position (m)")
    args = p.parse_args()

    prepare_urdf()
    cfg = load_robot_config()

    joint_names, positions, dt = plan(
        cfg,
        goal_world=args.goal,
        delta=args.delta,
    )

    if positions is not None:
        save(joint_names, positions, dt)


if __name__ == "__main__":
    main()
