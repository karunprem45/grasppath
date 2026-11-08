"""Run randomized pick-and-place trials and save one row per trial to results/trials.csv.

Each trial: a cube at a random spot and angle on one side of a wall, a random place
target on the other side. The arm has to plan around the wall.
"""
import argparse
import csv
import os
import numpy as np

from grasppath.sim import PandaScene, SimController
from grasppath.kinematics import PANDA_HOME
from grasppath.state_machine import PickPlaceFSM, State

FIELDS = ["trial", "success", "failed_state", "place_err_mm", "grasp_retries",
          "plan_time_s", "rrt_nodes", "ik_calls", "ik_fail", "path_len_raw",
          "path_len_short", "max_track_err", "sim_time_s",
          "cube_x", "cube_y", "cube_yaw_deg", "place_x", "place_y"]


def one_trial(scene, i, rng):
    scene.reset_arm(PANDA_HOME)
    scene.settle(30, PANDA_HOME)
    cx, cy = rng.uniform(0.40, 0.60), rng.uniform(0.15, 0.35)
    yaw = rng.uniform(-np.pi, np.pi)
    px, py = rng.uniform(0.40, 0.60), rng.uniform(-0.35, -0.15)
    scene.place_cube(cx, cy, yaw)
    ctl = SimController(scene, (px, py), rng)
    fsm = PickPlaceFSM(ctl)
    final = fsm.run()
    cube_pos, _ = scene.cube_pose()
    err_mm = float(np.linalg.norm(cube_pos[:2] - [px, py]) * 1000)
    row = {"trial": i, "success": int(final == State.DONE),
           "failed_state": fsm.failure.name if fsm.failure else "",
           "place_err_mm": round(err_mm, 2), "grasp_retries": fsm.retries,
           "cube_x": round(cx, 3), "cube_y": round(cy, 3),
           "cube_yaw_deg": round(np.degrees(yaw), 1),
           "place_x": round(px, 3), "place_y": round(py, 3)}
    row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in ctl.m.items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--out", default="results/trials.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    scene = PandaScene(gui=args.gui)
    # wall between the pick side (y > 0) and the place side (y < 0)
    scene.add_box(center=(0.5, 0.0, 0.25), half=(0.1, 0.02, 0.25))
    rows = []
    for i in range(args.trials):
        row = one_trial(scene, i, rng)
        rows.append(row)
        print(f"trial {i:3d}: success={row['success']} failed={row['failed_state'] or '-'} "
              f"place_err={row['place_err_mm']:.1f} mm plan={row['plan_time_s']:.2f}s")
    scene.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["success"]]
    print("\n--- summary ---")
    print(f"success rate: {len(ok)}/{len(rows)}")
    if ok:
        errs = np.array([r["place_err_mm"] for r in ok])
        print(f"placement error on successes (mm): median {np.median(errs):.1f}, max {errs.max():.1f}")
        print(f"mean planning time per trial (s): {np.mean([r['plan_time_s'] for r in rows]):.3f}")
        raw = sum(r["path_len_raw"] for r in rows)
        short = sum(r["path_len_short"] for r in rows)
        if raw > 0:
            print(f"shortcutting cut joint-space path length by {100 * (1 - short / raw):.1f}%")
    fails = {}
    for r in rows:
        if not r["success"]:
            fails[r["failed_state"]] = fails.get(r["failed_state"], 0) + 1
    if fails:
        print("failures by state:", fails)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
