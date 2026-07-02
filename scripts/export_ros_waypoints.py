#!/usr/bin/env python3
"""Export simulated robot trajectories as waypoint CSVs for ROS/Gazebo replay."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import run_pipeline
from src.config_loader import load_config


def main() -> int:
    config_path = ROOT / "configs" / "football_full.yaml"
    if len(sys.argv) > 1:
        config_path = ROOT / sys.argv[1]

    config = load_config(config_path)
    results, _ = run_pipeline(config, ROOT)

    out_dir = ROOT / config.output.results_dir / "ros_waypoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        path = out_dir / f"{result.method}_N{result.num_robots}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "method",
                    "num_robots",
                    "robot_id",
                    "waypoint_index",
                    "x_m",
                    "y_m",
                ],
            )
            writer.writeheader()
            for robot in result.robots:
                for idx, (x, y) in enumerate(robot.path):
                    writer.writerow(
                        {
                            "method": result.method,
                            "num_robots": result.num_robots,
                            "robot_id": robot.robot_id,
                            "waypoint_index": idx,
                            "x_m": x,
                            "y_m": y,
                        }
                    )
        print(f"saved {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
