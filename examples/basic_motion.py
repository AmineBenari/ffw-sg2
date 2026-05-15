#!/usr/bin/env python3
"""
basic_motion.py — First FFW-SG2 MuJoCo demo.

Loads the robot in the MuJoCo viewer and runs a simple bilateral arm motion:
both arms swing in sync using position-controlled joints.

Run from repo root:
    python3 examples/basic_motion.py
"""

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer

SCENE = Path(__file__).parent.parent / "robot/mujoco/scene_ffw_sg2.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    # Helper: resolve actuator name → ctrl index
    def ctrl(name: str) -> int:
        return model.actuator(name).id

    print(f"Loaded: {model.nbody} bodies, {model.nu} actuators")
    print("Actuators:")
    for i in range(model.nu):
        print(f"  [{i:2d}] {model.actuator(i).name}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()

        while viewer.is_running():
            t = time.time() - start

            # --- Arm motion ---
            # Joint1: shoulder yaw  (range ±3.14 rad)  — swing arms forward/back
            # Joint4: elbow pitch   (range -2.94..1.08) — bend elbows symmetrically
            swing  =  0.6 * math.sin(0.4 * t)          # ±0.6 rad, slow
            bend   = -0.4 * max(0.0, math.sin(0.4 * t))

            data.ctrl[ctrl("arm_l_joint1")] =  swing
            data.ctrl[ctrl("arm_r_joint1")] = -swing    # mirror

            data.ctrl[ctrl("arm_l_joint4")] = bend
            data.ctrl[ctrl("arm_r_joint4")] = bend

            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time pacing
            elapsed   = time.time() - start - t
            remaining = model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
