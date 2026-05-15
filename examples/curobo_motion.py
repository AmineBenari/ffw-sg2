#!/usr/bin/env python3
"""
curobo_motion.py — CuRobo motion planning for the FFW-SG2 left arm.

Plans a collision-free trajectory and saves it to robot/traj.npz.
Run this INSIDE the Docker container (requires GPU + CuRobo):

    docker compose run ffw-sg2-planner python3 -u examples/curobo_motion.py

Then visualise the result on the HOST (native OpenGL → GPU-accelerated):

    python3 examples/play_trajectory.py
"""

import re
from pathlib import Path

import numpy as np
import torch
import yaml

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState, GoalToolPose

REPO_ROOT  = Path(__file__).parent.parent
URDF_SRC   = REPO_ROOT / "robot/urdf/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"
URDF_OUT   = REPO_ROOT / "robot/urdf/ffw_sg2_processed.urdf"
ROBOT_CFG  = REPO_ROOT / "robot/curobo/ffw_sg2_left_arm.yml"
TRAJ_OUT   = REPO_ROOT / "robot/traj.npz"


# ── Step 1: Preprocess URDF ────────────────────────────────────────────────

def prepare_urdf() -> Path:
    """Replace package:// and broken file:// URIs so CuRobo can load the URDF."""
    content = URDF_SRC.read_text()
    pkg_root = str(REPO_ROOT / "robot/urdf")
    content = content.replace("package://ffw_description/", pkg_root + "/")
    # RealSense mesh uses an absolute ROS install path — replace with dummy box
    content = re.sub(
        r'<mesh filename="file:///[^"]*\.stl"/>',
        '<box size="0.02 0.02 0.02"/>',
        content,
    )
    URDF_OUT.write_text(content)
    print(f"[urdf] written to {URDF_OUT.name}")
    return URDF_OUT


# ── Step 2: Load CuRobo config with resolved paths ────────────────────────

def load_robot_config() -> dict:
    """Load YAML and resolve relative paths to absolute paths."""
    cfg = yaml.safe_load(ROBOT_CFG.read_text())
    kin = cfg["robot_cfg"]["kinematics"]
    kin["urdf_path"]       = str(REPO_ROOT / kin["urdf_path"])
    kin["asset_root_path"] = str(REPO_ROOT / kin["asset_root_path"])
    return cfg


# ── Step 3: Plan with CuRobo ───────────────────────────────────────────────

def plan(robot_cfg_dict: dict):
    """
    Build a planner, run FK to find the current EE pose,
    then plan to the first reachable goal from a set of 10 cm Cartesian deltas.
    Returns (joint_names, positions, dt) — positions is (T, dof), dt in seconds.
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
    print(f"[curobo] Home EE quaternion: {ee_quat.cpu().tolist()}")

    deltas = [
        ( 0.10,  0.00,  0.00),
        (-0.10,  0.00,  0.00),
        ( 0.00, -0.10,  0.00),
        ( 0.00,  0.00,  0.10),
    ]

    result = None
    for dx, dy, dz in deltas:
        goal_pos = ee_pos.clone()
        goal_pos[0] += dx
        goal_pos[1] += dy
        goal_pos[2] += dz
        print(f"[curobo] Trying goal: {goal_pos.cpu().tolist()}")
        goal = GoalToolPose(
            tool_frames=planner.tool_frames,
            position=goal_pos.reshape(1, 1, 1, 1, 3),
            quaternion=ee_quat.reshape(1, 1, 1, 1, 4),
        )
        result = planner.plan_pose(goal, q_home)
        if result is not None and result.success.any():
            print(f"[curobo] Goal accepted (dx={dx}, dy={dy}, dz={dz})")
            break
        print(f"[curobo] Goal rejected, trying next...")

    if result is None or not result.success.any():
        print("[curobo] Planning FAILED for all candidate goals.")
        return None, None, None

    traj = result.get_interpolated_plan()
    dt   = planner.trajopt_solver.config.interpolation_dt

    dof  = traj.position.shape[-1]
    pos  = traj.position.reshape(-1, dof)   # (T, dof)
    T    = pos.shape[0]
    print(f"[curobo] Success: {T} waypoints, {T * dt:.2f}s")

    traj_joint_names = (
        traj.joint_names
        if hasattr(traj, "joint_names") and traj.joint_names is not None
        else planner.joint_names
    )
    return traj_joint_names, pos, dt


# ── Step 4: Save trajectory ────────────────────────────────────────────────

def save(joint_names, positions: torch.Tensor, dt: float):
    np.savez(
        TRAJ_OUT,
        joint_names=np.array(joint_names),
        positions=positions.cpu().numpy(),
        dt=np.float64(dt),
    )
    print(f"[traj] saved to {TRAJ_OUT}")
    print(f"[traj] visualise with:  python3 examples/play_trajectory.py")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    prepare_urdf()
    robot_cfg_dict = load_robot_config()
    joint_names, positions, dt = plan(robot_cfg_dict)
    if positions is not None:
        save(joint_names, positions, dt)


if __name__ == "__main__":
    main()
