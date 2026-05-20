#!/usr/bin/env python3
"""
curobo_server.py — Persistent CuRobo planning server.

Loads CuRobo once, keeps it warm, and serves planning requests indefinitely.
Communicates with interactive_plan.py via robot/goal.json (shared filesystem).

At startup the server writes robot/server_info_<arm>.json with the home EE
position in CuRobo's frame.  interactive_plan.py reads this to compute the
true MuJoCo→CuRobo frame offset dynamically — no hardcoded constants.

Run INSIDE Docker:
    docker compose run ffw-sg2-planner python3 -u examples/curobo_server.py
    docker compose run ffw-sg2-planner python3 -u examples/curobo_server.py --arm right
    docker compose run ffw-sg2-planner python3 -u examples/curobo_server.py --orientation

Flags:
    --arm {left,right}   Which arm to plan for (default: left)
    --orientation        Enable full 6DOF planning (position + orientation).
                         interactive_plan.py must also be started with --orientation.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState, GoalToolPose

from curobo._src.solver.solver_ik_cfg      import IKSolverCfg
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlannerCfg

REPO_ROOT    = Path(__file__).parent.parent
URDF_SRC     = REPO_ROOT / "robot/urdf/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"
URDF_OUT     = REPO_ROOT / "robot/urdf/ffw_sg2_processed.urdf"

_CUROBO_TASK     = Path("/opt/curobo/curobo/content/configs/task")
IK_OPT_YAML      = _CUROBO_TASK / "ik/lbfgs_ik.yml"
TRAJOPT_OPT_YAML = _CUROBO_TASK / "trajopt/lbfgs_bspline_trajopt.yml"


def prepare_urdf():
    content  = URDF_SRC.read_text()
    pkg_root = str(REPO_ROOT / "robot/urdf")
    content  = content.replace("package://ffw_description/", pkg_root + "/")
    content  = re.sub(r'<mesh filename="file:///[^"]*\.stl"/>',
                      '<box size="0.02 0.02 0.02"/>', content)
    URDF_OUT.write_text(content)


def load_robot_config(cfg_path: Path) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    kin = cfg["robot_cfg"]["kinematics"]
    kin["urdf_path"]       = str(REPO_ROOT / kin["urdf_path"])
    kin["asset_root_path"] = str(REPO_ROOT / kin["asset_root_path"])
    return cfg


def _make_planner_cfg(robot_cfg: dict, orientation: bool) -> MotionPlannerCfg:
    ik_opt = yaml.safe_load(IK_OPT_YAML.read_text())
    tj_opt = yaml.safe_load(TRAJOPT_OPT_YAML.read_text())

    if orientation:
        ik_opt["rollout"]["cost_cfg"]["tool_pose_cfg"]["weight"] = [10000.0, 10000.0]
        tj_opt["rollout"]["cost_cfg"]["tool_pose_cfg"]["weight"] = [1000000.0, 1000000.0]
        orient_tol         = 0.1
        seed_orient_weight = 1.0
    else:
        ik_opt["rollout"]["cost_cfg"]["tool_pose_cfg"]["weight"] = [10000.0, 0.0]
        tj_opt["rollout"]["cost_cfg"]["tool_pose_cfg"]["weight"] = [1000000.0, 0.0]
        orient_tol         = 1.0
        seed_orient_weight = 0.0

    ik_cfg = IKSolverCfg.create(
        robot=robot_cfg,
        optimizer_configs=[ik_opt],
        seed_orientation_weight=seed_orient_weight,
        orientation_tolerance=orient_tol,
        position_tolerance=0.005,
    )
    trajopt_cfg = TrajOptSolverCfg.create(
        robot=robot_cfg,
        optimizer_configs=[tj_opt],
        orientation_tolerance=orient_tol,
        position_tolerance=0.005,
    )
    graph_cfg = PRMGraphPlannerCfg.create(robot=robot_cfg)
    return MotionPlannerCfg(
        ik_solver_config=ik_cfg,
        trajopt_solver_config=trajopt_cfg,
        graph_planner_config=graph_cfg,
    )


def build_planner(cfg_path: Path, orientation: bool) -> MotionPlanner:
    config  = _make_planner_cfg(load_robot_config(cfg_path), orientation)
    planner = MotionPlanner(config)
    mode    = "6DOF" if orientation else "position-only"
    print(f"[server] Warming up CuRobo ({mode})...")
    planner.warmup(enable_graph=True, num_warmup_iterations=3)
    return planner


def write_server_info(planner: MotionPlanner, arm: str, server_info_path: Path):
    """Compute home EE position in CuRobo frame and write to server_info_<arm>.json."""
    q_home = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )
    fk      = planner.kinematics.compute_kinematics(q_home)
    ee_pose = fk.tool_poses.get_link_pose(planner.tool_frames[0])
    ee_home = ee_pose.position.squeeze().cpu().tolist()

    info = {"ee_home_curobo": ee_home, "arm": arm}
    server_info_path.write_text(json.dumps(info))
    print(f"[server] EE home (CuRobo frame): {[f'{v:.4f}' for v in ee_home]}")
    print(f"[server] Wrote {server_info_path.name} — interactive_plan.py can now start.")


def _biased_ik_seeds(start_pos: torch.Tensor,
                     planner: MotionPlanner,
                     num_ik_seeds: int) -> torch.Tensor:
    """Return `num_ik_seeds` seeds clustered around the current joint state.

    Seed 0 is the exact current state.  The remaining seeds are Gaussian
    perturbations with σ=0.2 rad, clamped to joint limits.  Providing the
    full `num_ik_seeds` tensor prevents SeedManager from appending random
    seeds drawn from the full joint range, which is the root cause of
    large-swing trajectories for small target moves.
    """
    device = start_pos.device
    dof    = start_pos.shape[-1]
    jl     = planner.ik_solver.kinematics.get_joint_limits()
    lo     = jl.position[0].view(1, 1, dof)
    hi     = jl.position[1].view(1, 1, dof)

    noise  = 0.2 * torch.randn(1, num_ik_seeds - 1, dof, device=device)
    perturbed = torch.clamp(start_pos.view(1, 1, dof) + noise, lo, hi)
    return torch.cat([start_pos.view(1, 1, dof), perturbed], dim=1)  # [1, N, dof]


def plan_to(planner: MotionPlanner,
            goal_pos_curobo: list[float],
            joint_state: dict | None = None,
            goal_quat: list[float] | None = None):
    """Plan to goal_pos_curobo which is already in CuRobo's coordinate frame.

    Strategy (two-pass):
      Pass 1 — local IK: seeds the IK with small perturbations of the current
        joint state so the planner stays in the same arm configuration (no
        elbow/wrist flips).  Fast and produces short, efficient trajectories.
      Pass 2 — global fallback: if the local IK fails (target genuinely
        requires a configuration change, or is near a singularity), falls back
        to plan_pose() with its full random-seed search.

    joint_state : {joint_name: rad} — rounded to 1 mrad to suppress physics noise.
    goal_quat   : [w, x, y, z] — ignored when server built without --orientation.
    """
    device = planner.default_joint_state.position.device

    goal_t = torch.tensor(goal_pos_curobo, dtype=torch.float32, device=device)

    if joint_state is not None:
        raw       = np.array([joint_state.get(n, 0.0) for n in planner.joint_names])
        start_pos = torch.tensor(
            np.round(raw, 3).tolist(), dtype=torch.float32, device=device
        ).unsqueeze(0)
    else:
        start_pos = planner.default_joint_state.position.unsqueeze(0)

    q_start = JointState.from_position(start_pos, joint_names=planner.joint_names)

    ee_quat = torch.tensor(
        goal_quat if goal_quat is not None else [1.0, 0.0, 0.0, 0.0],
        dtype=torch.float32, device=device,
    )

    goal = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=goal_t.reshape(1, 1, 1, 1, 3),
        quaternion=ee_quat.reshape(1, 1, 1, 1, 4),
    )

    # ── Pass 1: local IK with biased seeds ───────────────────────────────────
    num_ik_seeds   = planner.ik_solver.config.num_seeds        # 32
    num_traj_seeds = planner.trajopt_solver.config.num_seeds   # 4
    result = None

    try:
        biased = _biased_ik_seeds(start_pos, planner, num_ik_seeds)
        ik_result = planner.ik_solver.solve_pose(
            goal,
            current_state=q_start,
            seed_config=biased,
            return_seeds=num_traj_seeds,
        )

        if ik_result.success.any():
            # Fill any failed seeds with the best successful solution
            seed_cfg = ik_result.solution  # [1, num_traj_seeds, dof]
            if not ik_result.success.all():
                flat     = seed_cfg.view(-1, seed_cfg.shape[-1])
                good     = flat[ik_result.success.view(-1)][0:1].clone()
                flat[~ik_result.success.view(-1)] = good

            traj_result = planner.trajopt_solver.solve_pose(
                goal, q_start,
                seed_config=seed_cfg,
                use_implicit_goal=True,
                finetune_attempts=1,
                finetune_dt_scale=0.55,
            )
            if traj_result is not None and traj_result.success.any():
                result = traj_result
                print("[server] local IK succeeded")
    except Exception as e:
        print(f"[server] local IK error: {e} — falling back")

    # ── Pass 2: global fallback (full random-seed search) ────────────────────
    if result is None:
        print("[server] local IK failed — retrying with global random seeds")
        result = planner.plan_pose(goal, q_start)

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


def _write_goal(goal: dict, goal_file: Path):
    tmp = goal_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(goal))
    tmp.rename(goal_file)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["left", "right"], default="left")
    p.add_argument("--orientation", action="store_true",
                   help="Enable full 6DOF planning (position + orientation).")
    args = p.parse_args()

    goal_file   = REPO_ROOT / f"robot/goal_{args.arm}.json"
    traj_out    = REPO_ROOT / f"robot/traj_{args.arm}.npz"
    server_info = REPO_ROOT / f"robot/server_info_{args.arm}.json"

    cfg_file = REPO_ROOT / "robot/curobo" / f"ffw_sg2_{args.arm}_arm.yml"
    print(f"[server] Arm: {args.arm}  Config: {cfg_file.name}")

    prepare_urdf()
    planner = build_planner(cfg_file, args.orientation)
    write_server_info(planner, args.arm, server_info)
    print(f"[server] Ready — waiting for goals in robot/goal_{args.arm}.json\n")

    last_id = None
    while True:
        time.sleep(0.05)
        if not goal_file.exists():
            continue
        try:
            goal = json.loads(goal_file.read_text())
        except Exception:
            continue

        if goal.get("status") != "pending":
            continue
        req_id = goal.get("id")
        if req_id == last_id:
            continue

        last_id          = req_id
        goal_pos_curobo  = goal["position"]   # already in CuRobo frame
        joint_state      = goal.get("joint_state")
        goal_quat        = goal.get("orientation")

        start_desc   = "current state" if joint_state else "home"
        orient_desc  = f"  orient={[f'{v:.3f}' for v in goal_quat]}" if goal_quat else ""
        print(f"[server] Planning from {start_desc} → "
              f"{[f'{v:.3f}' for v in goal_pos_curobo]}{orient_desc}  (id={req_id})")

        goal["status"] = "planning"
        _write_goal(goal, goal_file)

        names, pos, dt = plan_to(planner, goal_pos_curobo,
                                  joint_state=joint_state,
                                  goal_quat=goal_quat)

        if pos is not None:
            np.savez(traj_out,
                     joint_names=np.array(names),
                     positions=pos.cpu().numpy(),
                     dt=np.float64(dt))
            goal["status"] = "done"
            print(f"[server] Done: {pos.shape[0]} waypoints ({pos.shape[0]*dt:.2f}s)")
        else:
            goal["status"] = "failed"
            print("[server] Planning failed — goal unreachable or in collision")

        _write_goal(goal, goal_file)


if __name__ == "__main__":
    main()
