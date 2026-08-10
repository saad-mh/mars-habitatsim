#!/usr/bin/env python3
"""Compare pure NavDP, S2Diff, and HLC-S2Diff rollout archives."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_run(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("run must be LABEL=/path/to/rollout.npz")
    label, path = specification.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("run label cannot be empty")
    return label, Path(path).expanduser().resolve()


def optional_array(data: Any, key: str, length: int, default: float = 0.0) -> np.ndarray:
    if key not in data:
        return np.full(length, default, dtype=np.float64)
    return np.asarray(data[key], dtype=np.float64).reshape(-1)


def longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def summarize(path: Path, deadlock_speed: float, deadlock_frames: int) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        actions = np.asarray(data["action_3d"], dtype=np.float64)
        poses = np.asarray(data["pose"], dtype=np.float64)
        goal_distance = np.asarray(data["goal_distance"], dtype=np.float64).reshape(-1)
        length = len(goal_distance)
        hz = float(np.asarray(data.get("hz", 10.0)))
        success = bool(np.asarray(data.get("success", False)))
        forward_speed = np.abs(actions[:, 0])
        yaw_rate = np.abs(actions[:, 2])
        stopped = (forward_speed < deadlock_speed) & (yaw_rate < deadlock_speed)
        fallback = optional_array(data, "fallback_stop", length).astype(bool)
        escape = optional_array(data, "escape_turn", length).astype(bool)
        signs = optional_array(data, "selected_circulation_sign", length)
        active_signs = signs[np.abs(signs) > 0.5]
        sign_switches = int(np.sum(active_signs[1:] != active_signs[:-1]))
        latency = optional_array(data, "planning_time_seconds", length, np.nan)
        latency = latency[np.isfinite(latency)]
        clearance = optional_array(data, "selected_minimum_clearance", length, np.nan)
        clearance = clearance[np.isfinite(clearance) & (clearance >= 0.0)]

        positions = poses[:, [0, 2]]
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        initial_goal_distance = float(goal_distance[0])
        euclidean_spl = (
            initial_goal_distance / max(initial_goal_distance, path_length, 1.0e-8)
            if success
            else 0.0
        )
        longest_stop = longest_true_run(stopped)
        deadlock = longest_stop >= deadlock_frames and not success

        return {
            "success": float(success),
            "final_goal_distance_m": float(goal_distance[-1]),
            "goal_progress_m": initial_goal_distance - float(goal_distance[-1]),
            "path_length_m": path_length,
            "euclidean_spl": euclidean_spl,
            "duration_s": length / max(hz, 1.0e-8),
            "mean_forward_speed_mps": float(forward_speed.mean()),
            "stopped_fraction": float(stopped.mean()),
            "longest_stop_s": longest_stop / max(hz, 1.0e-8),
            "deadlock": float(deadlock),
            "fallback_count": float(fallback.sum()),
            "escape_count": float(escape.sum()),
            "mode_switch_count": float(sign_switches),
            "minimum_clearance_m": float(clearance.min()) if clearance.size else np.nan,
            "mean_clearance_m": float(clearance.mean()) if clearance.size else np.nan,
            "latency_p50_ms": float(np.percentile(latency, 50) * 1000.0)
            if latency.size
            else np.nan,
            "latency_p95_ms": float(np.percentile(latency, 95) * 1000.0)
            if latency.size
            else np.nan,
        }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label"])].append(row)
    output = []
    for label, group in grouped.items():
        combined: dict[str, Any] = {"label": label, "episodes": len(group)}
        for key in group[0]:
            if key in {"label", "path"}:
                continue
            values = np.asarray([item[key] for item in group], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                combined[f"{key}_mean"] = np.nan
                combined[f"{key}_std"] = np.nan
            else:
                combined[f"{key}_mean"] = float(finite.mean())
                combined[f"{key}_std"] = (
                    float(finite.std(ddof=1)) if finite.size > 1 else 0.0
                )
        output.append(combined)
    return output


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("label", "model"),
        ("episodes", "N"),
        ("success_mean", "success"),
        ("deadlock_mean", "deadlock"),
        ("minimum_clearance_m_mean", "min_clear"),
        ("stopped_fraction_mean", "stop_frac"),
        ("euclidean_spl_mean", "eSPL"),
        ("latency_p50_ms_mean", "p50_ms"),
        ("latency_p95_ms_mean", "p95_ms"),
    ]
    print("  ".join(f"{title:>11}" for _, title in columns))
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, np.nan)
            values.append(f"{value:>11}" if isinstance(value, str) else f"{value:11.4f}")
        print("  ".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--deadlock-speed", type=float, default=0.03)
    parser.add_argument("--deadlock-seconds", type=float, default=1.0)
    parser.add_argument("--output", default="runs/ablation_summary")
    args = parser.parse_args()

    rows = []
    for label, path in args.run:
        with np.load(path, allow_pickle=False) as data:
            hz = float(np.asarray(data.get("hz", 10.0)))
        metrics = summarize(
            path,
            deadlock_speed=args.deadlock_speed,
            deadlock_frames=max(1, int(round(args.deadlock_seconds * hz))),
        )
        rows.append({"label": label, "path": str(path), **metrics})
    summary = aggregate(rows)
    print_table(summary)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump({"episodes": rows, "summary": summary}, stream, indent=2)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved {output.with_suffix('.json')}")
    print(f"Saved {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
