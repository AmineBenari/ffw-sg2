#!/usr/bin/env python3
"""
play_trajectory.py — Replay a CuRobo trajectory in MuJoCo.

Loads robot/traj.npz (produced by curobo_motion.py) and plays it back
in MuJoCo with full physics and a GPU-accelerated viewer.

Run on the HOST (not inside Docker) so the viewer gets native OpenGL:
    python3 examples/play_trajectory.py
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
SCENE_XML = REPO_ROOT / "robot/mujoco/scene_ffw_sg2.xml"
TRAJ_FILE = REPO_ROOT / "robot/traj.npz"


def main():
    # ── Load trajectory ────────────────────────────────────────────────────
    data_npz     = np.load(TRAJ_FILE, allow_pickle=True)
    joint_names  = list(data_npz["joint_names"])
    positions    = data_npz["positions"]   # (T, dof)
    curobo_dt    = float(data_npz["dt"])
    T            = positions.shape[0]
    print(f"[traj] {T} waypoints × {curobo_dt*1000:.1f} ms = {T*curobo_dt:.2f}s")
    print(f"[traj] joints: {joint_names}")

    # ── Load MuJoCo model ─────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data  = mujoco.MjData(model)
    mujoco_dt = model.opt.timestep  # typically 0.002 s

    # Map joint names → actuator ctrl indices
    ctrl_idx: dict[str, int] = {}
    for i in range(model.nu):
        name = model.actuator(i).name
        if name in joint_names:
            ctrl_idx[name] = i

    print(f"[mujoco] Mapped {len(ctrl_idx)} actuators: {list(ctrl_idx)}")

    # Physics steps per CuRobo waypoint — keeps sim time aligned with plan time
    steps_per_wpt = max(1, round(curobo_dt / mujoco_dt))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = 3.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 135

        print(f"[mujoco] Playing trajectory. Close viewer to exit.")

        wall_start = time.perf_counter()

        for t in range(T):
            if not viewer.is_running():
                return

            # Set position targets for this waypoint
            for j, name in enumerate(joint_names):
                if name in ctrl_idx:
                    data.ctrl[ctrl_idx[name]] = positions[t, j]

            # Run enough physics steps to cover curobo_dt of simulation time
            for _ in range(steps_per_wpt):
                mujoco.mj_step(model, data)

            viewer.sync()

            # Real-time pacing — sleep until wall clock catches up to sim time
            sim_elapsed  = (t + 1) * curobo_dt
            wall_elapsed = time.perf_counter() - wall_start
            slack = sim_elapsed - wall_elapsed
            if slack > 0:
                time.sleep(slack)

        # ── Hold final pose ────────────────────────────────────────────────
        print("[mujoco] Holding final pose.")
        while viewer.is_running():
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed   = time.perf_counter() - step_start
            remaining = mujoco_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
