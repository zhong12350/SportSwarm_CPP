#!/usr/bin/env python3
"""Generate synthetic StatsBomb-style out-of-bounds events for BLF testing."""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "statsbomb_sample" / "events.json"

# StatsBomb pitch 120 x 80; out events cluster near touchlines
HOTSPOTS = [
    (118, 15, "offensive"),
    (118, 65, "offensive"),
    (2, 40, "defensive"),
    (60, 2, "neutral"),
    (60, 78, "neutral"),
]


def main() -> None:
    random.seed(42)
    events = []
    for i in range(300):
        hx, hy, style = random.choice(HOTSPOTS)
        x = hx + random.gauss(0, 4)
        y = hy + random.gauss(0, 6)
        x = max(0, min(120, x))
        y = max(0, min(80, y))
        etype = "Out" if random.random() > 0.15 else "Miscontrol"
        if style == "offensive":
            pass_type = random.choice(["Pass", "Cross", "Shot"])
        elif style == "defensive":
            pass_type = random.choice(["Clearance", "Block"])
        else:
            pass_type = "Pass"
        events.append(
            {
                "id": i,
                "type": {"name": etype},
                "pass": {"type": {"name": pass_type}},
                "location": [round(x, 2), round(y, 2)],
                "out": True,
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)
    print(f"Wrote {len(events)} events to {OUT_PATH}")


if __name__ == "__main__":
    main()
