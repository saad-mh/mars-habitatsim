#!/usr/bin/env python3
"""Paper-grade analysis for pure NavDP, S2Diff, and HLC-S2Diff rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METADATA_KEYS = {"label", "path", "layout", "seed", "clearance_source"}
PAIRED_METRICS = {
    "success": True,
    "collision": False,
    "deadlock": False,
    "minimum_clearance_m": True,
    "mean_clearance_m": True,
    "path_efficiency": True,
    "euclidean_spl": True,
    "stopped_fraction": False,
    "mode_switch_count": False,
    "latency_p50_ms": False,
    "latency_p95_ms": False,
}


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


def scalar_value(data: Any, key: str, default: Any) -> Any:
    if key not in data:
        return default
    value = np.asarray(data[key])
    return value.item() if value.ndim == 0 else value.reshape(-1)[0].item()


def infer_seed(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.fullmatch(r"seed_(-?\d+)", part)
        if match:
            return int(match.group(1))
    return -1


def infer_layout(path: Path) -> str:
    # Paper runner layout: .../<model>/<layout>/seed_<n>/rollout.npz
    if len(path.parents) >= 3 and path.parent.name.startswith("seed_"):
        return path.parent.parent.name
    return "legacy"


def longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def summarize(path: Path, deadlock_speed: float, deadlock_frames: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        actions = np.asarray(data["action_3d"], dtype=np.float64)
        poses = np.asarray(data["pose"], dtype=np.float64)
        goal_distance = np.asarray(data["goal_distance"], dtype=np.float64).reshape(-1)
        length = len(goal_distance)
        if length == 0:
            raise ValueError(f"empty rollout: {path}")

        hz = float(scalar_value(data, "hz", 10.0))
        success = bool(scalar_value(data, "success", False))
        layout = str(scalar_value(data, "evaluation_layout", infer_layout(path)))
        seed = int(scalar_value(data, "seed", infer_seed(path)))
        stop_distance = float(scalar_value(data, "stop_distance", 1.0))

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

        if "executed_surface_clearance" in data:
            clearance_source = "executed_mesh_surface"
            clearance = optional_array(data, "executed_surface_clearance", length, np.nan)
        else:
            clearance_source = "legacy_planner_prediction"
            clearance = optional_array(data, "selected_minimum_clearance", length, np.nan)
        clearance = clearance[np.isfinite(clearance) & (clearance >= 0.0)]
        guidance_correction = optional_array(
            data, "mean_guidance_noise_correction", length, np.nan
        )
        guidance_correction = guidance_correction[np.isfinite(guidance_correction)]
        effective_sample_size = optional_array(
            data, "mean_final_effective_sample_size", length, np.nan
        )
        effective_sample_size = effective_sample_size[
            np.isfinite(effective_sample_size)
        ]

        collisions = optional_array(data, "geometric_collision", length).astype(bool)
        collision = bool(collisions.any())

        positions = poses[:, [0, 2]]
        if "start_position_xz" in data:
            start = np.asarray(data["start_position_xz"], dtype=np.float64).reshape(1, 2)
            positions = np.concatenate((start, positions), axis=0)
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())

        initial_goal_distance = float(
            scalar_value(data, "initial_goal_distance", float(goal_distance[0]))
        )
        final_goal_distance = float(goal_distance[-1])
        goal_progress = initial_goal_distance - final_goal_distance
        path_efficiency = (
            float(np.clip(max(goal_progress, 0.0) / path_length, 0.0, 1.0))
            if path_length > 1.0e-8
            else 0.0
        )
        required_success_travel = max(initial_goal_distance - stop_distance, 0.0)
        euclidean_spl = (
            required_success_travel
            / max(required_success_travel, path_length, 1.0e-8)
            if success
            else 0.0
        )

        longest_stop = longest_true_run(stopped)
        deadlock = longest_stop >= deadlock_frames and not success

        return {
            "layout": layout,
            "seed": seed,
            "clearance_source": clearance_source,
            "success": float(success),
            "collision": float(collision),
            "collision_fraction": float(collisions.mean()),
            "final_goal_distance_m": final_goal_distance,
            "goal_progress_m": goal_progress,
            "path_length_m": path_length,
            "path_efficiency": path_efficiency,
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
            "mean_guidance_correction": (
                float(guidance_correction.mean()) if guidance_correction.size else np.nan
            ),
            "mean_particle_ess": (
                float(effective_sample_size.mean())
                if effective_sample_size.size
                else np.nan
            ),
            "latency_p50_ms": (
                float(np.percentile(latency, 50) * 1000.0) if latency.size else np.nan
            ),
            "latency_p95_ms": (
                float(np.percentile(latency, 95) * 1000.0) if latency.size else np.nan
            ),
        }


def bootstrap_mean_ci(
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1 or samples <= 0:
        value = float(values.mean())
        return value, value
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def aggregate(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label"])].append(row)

    rng = np.random.default_rng(bootstrap_seed)
    output: list[dict[str, Any]] = []
    for label, group in grouped.items():
        combined: dict[str, Any] = {
            "label": label,
            "episodes": len(group),
            "layouts": len({str(item["layout"]) for item in group}),
        }
        metric_keys = [key for key in group[0] if key not in METADATA_KEYS]
        for key in metric_keys:
            values = np.asarray([item[key] for item in group], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                mean = std = low = high = np.nan
            else:
                mean = float(finite.mean())
                std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
                layout_means = []
                for layout in sorted({str(item["layout"]) for item in group}):
                    layout_values = np.asarray(
                        [item[key] for item in group if str(item["layout"]) == layout],
                        dtype=np.float64,
                    )
                    layout_values = layout_values[np.isfinite(layout_values)]
                    if layout_values.size:
                        layout_means.append(float(layout_values.mean()))
                low, high = bootstrap_mean_ci(
                    np.asarray(layout_means), bootstrap_samples, rng
                )
            combined[f"{key}_mean"] = mean
            combined[f"{key}_std"] = std
            combined[f"{key}_ci95_low"] = low
            combined[f"{key}_ci95_high"] = high
        output.append(combined)
    return output


def paired_effects(
    rows: list[dict[str, Any]],
    reference_label: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    by_label: dict[str, dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair = (str(row["layout"]), int(row["seed"]))
        by_label[str(row["label"])][pair] = row
    if reference_label not in by_label:
        return []

    rng = np.random.default_rng(bootstrap_seed + 1)
    reference = by_label[reference_label]
    output: list[dict[str, Any]] = []
    for comparator_label, comparator in by_label.items():
        if comparator_label == reference_label:
            continue
        common_pairs = sorted(set(reference) & set(comparator))
        for metric, higher_is_better in PAIRED_METRICS.items():
            advantages = []
            advantages_by_layout: dict[str, list[float]] = defaultdict(list)
            for pair in common_pairs:
                reference_value = float(reference[pair][metric])
                comparator_value = float(comparator[pair][metric])
                if not np.isfinite(reference_value) or not np.isfinite(comparator_value):
                    continue
                raw_difference = reference_value - comparator_value
                advantage = raw_difference if higher_is_better else -raw_difference
                advantages.append(advantage)
                advantages_by_layout[pair[0]].append(advantage)
            values = np.asarray(advantages, dtype=np.float64)
            layout_values = np.asarray(
                [np.mean(items) for items in advantages_by_layout.values()],
                dtype=np.float64,
            )
            if layout_values.size:
                low, high = bootstrap_mean_ci(
                    layout_values, bootstrap_samples, rng
                )
                mean = float(layout_values.mean())
            else:
                mean = low = high = np.nan
            output.append(
                {
                    "reference": reference_label,
                    "comparator": comparator_label,
                    "metric": metric,
                    "positive_means_reference_better": True,
                    "paired_episodes": int(values.size),
                    "paired_layouts": int(layout_values.size),
                    "reference_advantage_mean": mean,
                    "reference_advantage_ci95_low": low,
                    "reference_advantage_ci95_high": high,
                }
            )
    return output


def json_compatible(value: Any) -> Any:
    """Replace non-finite numbers with JSON-standard null values."""
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return json_compatible(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("label", "model"),
        ("episodes", "N"),
        ("layouts", "layouts"),
        ("success_mean", "success"),
        ("collision_mean", "collision"),
        ("deadlock_mean", "deadlock"),
        ("minimum_clearance_m_mean", "min_clear"),
        ("path_efficiency_mean", "path_eff"),
        ("latency_p50_ms_mean", "p50_ms"),
        ("mean_guidance_correction_mean", "guide_rms"),
        ("mean_particle_ess_mean", "ESS"),
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
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument("--reference", default="hlc_full_k4")
    parser.add_argument("--output", default="runs/ablation_summary")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for label, path in args.run:
        with np.load(path, allow_pickle=False) as data:
            hz = float(scalar_value(data, "hz", 10.0))
        metrics = summarize(
            path,
            deadlock_speed=args.deadlock_speed,
            deadlock_frames=max(1, int(round(args.deadlock_seconds * hz))),
        )
        rows.append({"label": label, "path": str(path), **metrics})

    summary = aggregate(rows, args.bootstrap_samples, args.bootstrap_seed)
    effects = paired_effects(
        rows,
        reference_label=args.reference,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print_table(summary)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json_compatible(
        {
            "reference": args.reference,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "layout-level means",
            "episodes": rows,
            "summary": summary,
            "paired_effects": effects,
        }
    )
    with output.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    write_csv(output.with_suffix(".csv"), summary)
    write_csv(output.with_name(output.name + "_episodes.csv"), rows)
    write_csv(output.with_name(output.name + "_paired.csv"), effects)

    print(f"Saved {output.with_suffix('.json')}")
    print(f"Saved {output.with_suffix('.csv')}")
    print(f"Saved {output.with_name(output.name + '_episodes.csv')}")
    if effects:
        print(f"Saved {output.with_name(output.name + '_paired.csv')}")


if __name__ == "__main__":
    main()