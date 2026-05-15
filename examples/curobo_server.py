#!/usr/bin/env python3
"""
curobo_server.py — Persistent CuRobo planning server.

Loads CuRobo once, keeps it warm, and serves planning requests indefinitely.
Communicates with interactive_plan.py via robot/goal.json (shared filesystem).

Run INSIDE Docker:
    docker compose run ffw-sg2-planner python3 -u examples/curobo_server.py
"""

import json
import re
import time
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
GOAL_FILE = REPO_ROOT / "robot/goal.json"
TRAJ_OUT  = REPO_ROOT / "robot/traj.npz"


def prepare_urdf():
    content  = URDF_SRC.read_text()
    pkg_root = str(REPO_ROOT / "robot/urdf")
    content  = content.replace("package://ffw_description/", pkg_root + "/")
    content  = re.sub(r'<mesh filename="file:///[^"]*\.stl"/>',
                      '<box size="0.02 0.02 0.02"/>', content)
    URDF_OUT.write_text(content)


def load_robot_config():
    cfg = yaml.safe_load(ROBOT_CFG.read_text())
    kin = cfg["robot_cfg"]["kinematics"]
    kin["urdf_path"]       = str(REPO_ROOT / kin["urdf_path"])
    kin["asset_root_path"] = str(REPO_ROOT / kin["asset_root_path"])
    return cfg


def build_planner() -> MotionPlanner:
    config  = MotionPlannerCfg.create(robot=load_robot_config())
    planner = MotionPlanner(config)
    print("[server] Warming up CuRobo (first run compiles CUDA graphs)...")
    planner.warmup(enable_graph=True, num_warmup_iterations=3)
    print("[server] Ready — waiting for goals in robot/goal.json")
    print("[server] Start interactive_plan.py on the host to begin.")
    return planner


def plan_to(planner: MotionPlanner, goal_pos: list[float]):
    device   = planner.default_joint_state.position.device
    goal_t   = torch.tensor(goal_pos, dtype=torch.float32, device=device)

    q_home = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )
    fk      = planner.kinematics.compute_kinematics(q_home)
    ee_quat = fk.tool_poses.get_link_pose(planner.tool_frames[0]).quaternion.squeeze()

    goal = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=goal_t.reshape(1, 1, 1, 1, 3),
        quaternion=ee_quat.reshape(1, 1, 1, 1, 4),
    )
    result = planner.plan_pose(goal, q_home)

    if result is None or not result.success.any():
        return None, None, None

    traj  = result.get_interpolated_plan()
    dt    = planner.trajopt_solver.config.interpolation_dt
    dof   = traj.position.shape[-1]
    pos   = traj.position.reshape(-1, dof)
    names = (traj.joint_names
             if hasattr(traj, "joint_names") and traj.joint_names is not None
             else planner.joint_names)
    return names, pos, dt


def _write_goal(goal: dict):
    """Atomic write — rename avoids partial reads on the client side."""
    tmp = GOAL_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(goal))
    tmp.rename(GOAL_FILE)


def main():
    prepare_urdf()
    planner = build_planner()
    last_id = None

    while True:
        time.sleep(0.05)

        if not GOAL_FILE.exists():
            continue
        try:
            goal = json.loads(GOAL_FILE.read_text())
        except Exception:
            continue

        if goal.get("status") != "pending":
            continue
        req_id = goal.get("id")
        if req_id == last_id:
            continue

        last_id   = req_id
        goal_pos  = goal["position"]
        print(f"[server] Planning to {[f'{v:.3f}' for v in goal_pos]}  (id={req_id})")

        goal["status"] = "planning"
        _write_goal(goal)

        names, pos, dt = plan_to(planner, goal_pos)

        if pos is not None:
            np.savez(TRAJ_OUT,
                     joint_names=np.array(names),
                     positions=pos.cpu().numpy(),
                     dt=np.float64(dt))
            goal["status"] = "done"
            print(f"[server] Done: {pos.shape[0]} waypoints ({pos.shape[0]*dt:.2f}s)")
        else:
            goal["status"] = "failed"
            print("[server] Planning failed — goal unreachable or in collision")

        _write_goal(goal)


if __name__ == "__main__":
    main()
