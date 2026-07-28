#!/usr/bin/env python3

"""
Run every sigma_min_sweep experiment sequentially.

Run from the repository root:

    conda run -n sam2 --no-capture-output python exp.py

Outputs are written to

belief_exp/results/
    experiment_name_summary.csv
    experiment_name_details.csv
    experiment_name.log
"""

from pathlib import Path
import subprocess
import time
import sys

prev = time.time()

RESULTS = Path("belief_exp/results")
RESULTS.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable
SCRIPT = "belief_exp/sigma_min_sweep.py"

seed = 10


def calculate_time(now):
    elapsed = now - prev
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


EXPERIMENTS = [
    (
        "01_smoke_test",
        [
            "--episodes-per-config",
            "5",
            "--sigma-visible-grid",
            "1e-3",
            "0.2",
            "4",
            "--odom-noise-grid",
            "1e-3",
            "0.1",
            "4",
            "--other-grid-n",
            "2",
            "--params",
            "decay_factor,success_radius",
            "--seed",
            str(seed),
        ],
    ),
    (
        "02_env_odom_noise",
        [
            "--params",
            "env_odom_noise",
            "--env-odom-noise-range",
            "0.0",
            "0.15",
            "--other-grid-n",
            "8",
            "--sigma-visible-grid",
            "1e-4",
            "0.5",
            "12",
            "--odom-noise-grid",
            "1e-4",
            "0.3",
            "12",
            "--episodes-per-config",
            "20",
            "--seed",
            str(seed),
        ],
    ),
    (
        "03_env_odom_noise_wide",
        [
            "--params",
            "env_odom_noise",
            "--env-odom-noise-range",
            "0.0",
            "0.4",
            "--other-grid-n",
            "10",
            "--sigma-visible-grid",
            "1e-4",
            "0.5",
            "12",
            "--odom-noise-grid",
            "1e-4",
            "0.3",
            "12",
            "--episodes-per-config",
            "20",
            "--seed",
            str(seed),
        ],
    ),
    (
        "04_zoom_boundary",
        [
            "--params",
            "env_odom_noise",
            "--env-odom-noise-range",
            "0.0",
            "0.15",
            "--other-grid-n",
            "8",
            "--sigma-visible-grid",
            "0.005",
            "0.1",
            "16",
            "--odom-noise-grid",
            "0.005",
            "0.08",
            "16",
            "--episodes-per-config",
            "30",
            "--seed",
            str(seed),
        ],
    ),
    (
        "05_cross_sensitivity",
        [
            "--params",
            "env_odom_noise,env_obs_noise",
            "--other-grid-n",
            "6",
            "--episodes-per-config",
            "20",
            "--seed",
            str(seed),
        ],
    ),
    (
        "06_seed1",
        [
            "--params",
            "env_odom_noise",
            "--seed",
            str(seed),
            "--episodes-per-config",
            "20",
        ],
    ),
    (
        "07_strict_viability",
        [
            "--params",
            "env_odom_noise",
            "--coverage-dev-tol",
            "0.10",
            "--min-advance-rate",
            "0.8",
            "--max-false-advance-rate",
            "0.05",
            "--seed",
            str(seed),
        ],
    ),
    (
        "08_env_obs_noise",
        [
            "--params",
            "env_obs_noise",
            "--env-obs-noise-range",
            "0.02",
            "0.30",
            "--other-grid-n",
            "8",
            "--sigma-visible-grid",
            "1e-4",
            "0.5",
            "12",
            "--odom-noise-grid",
            "1e-4",
            "0.3",
            "12",
            "--episodes-per-config",
            "20",
            "--seed",
            str(seed),
        ],
    ),
    (
        "09_decay_factor",
        [
            "--params",
            "decay_factor",
            "--decay-factor-range",
            "0.80",
            "0.999",
            "--other-grid-n",
            "10",
            "--episodes-per-config",
            "20",
            "--seed",
            str(seed),
        ],
    ),
    (
        "10_success_radius",
        [
            "--params",
            "success_radius",
            "--success-radius-range",
            "0.2",
            "1.0",
            "--other-grid-n",
            "10",
            "--episodes-per-config",
            "20",
            "--seed",
            str(seed),
        ],
    ),
    (
        "11_gate_threshold",
        [
            "--params",
            "gate_threshold",
            "--gate-threshold-range",
            "0.1",
            "2.0",
            "--other-grid-n",
            "10",
            "--episodes-per-config",
            "20",
            "--seed",
            str(seed),
        ],
    ),
    (
        "12_sigma_init",
        [
            "--params",
            "sigma_init",
            "--sigma-init-range",
            "0.1",
            "5.0",
            "--other-grid-n",
            "10",
            "--episodes-per-config",
            "20",
            "--seed",
            str(seed),
        ],
    ),
    (
        "13_large_uncertainty",
        [
            "--params",
            "large_uncertainty",
            "--large-uncertainty-range",
            "50",
            "5000",
            "--other-grid-n",
            "10",
            "--episodes-per-config",
            "20",
            "--seed",
            str(seed),
        ],
    ),
    (
        "14_high_accuracy",
        [
            "--params",
            "env_odom_noise",
            "--env-odom-noise-range",
            "0.0",
            "0.15",
            "--other-grid-n",
            "8",
            "--sigma-visible-grid",
            "0.01",
            "0.05",
            "32",
            "--odom-noise-grid",
            "0.01",
            "0.05",
            "32",
            "--episodes-per-config",
            "100",
            "--seed",
            str(seed),
        ],
    ),
    (
        "15_fine_boundary",
        [
            "--params",
            "env_odom_noise",
            "--env-odom-noise-range",
            "0.04",
            "0.08",
            "--other-grid-n",
            "8",
            "--sigma-visible-grid",
            "0.01",
            "0.03",
            "40",
            "--odom-noise-grid",
            "0.01",
            "0.03",
            "40",
            "--episodes-per-config",
            "50",
            "--seed",
            str(seed),
        ],
    ),
    (
        "16_long_episodes",
        [
            "--params",
            "env_odom_noise",
            "--env-odom-noise-range",
            "0.0",
            "0.15",
            "--other-grid-n",
            "8",
            "--max-steps",
            "80",
            "--episodes-per-config",
            "20",
            "--seed",
            str(seed),
        ],
    ),
    (
        "17_large_diversity",
        [
            "--params",
            "env_odom_noise",
            "--env-odom-noise-range",
            "0.0",
            "0.15",
            "--other-grid-n",
            "20",
            "--episodes-per-config",
            "20",
            "--seed",
            str(seed),
        ],
    ),
    (
        "18_conservative",
        [
            "--params",
            "env_odom_noise",
            "--coverage-dev-tol",
            "0.05",
            "--min-advance-rate",
            "0.90",
            "--max-false-advance-rate",
            "0.02",
            "--episodes-per-config",
            "30",
            "--seed",
            str(seed),
        ],
    ),
    (
        "19_relaxed",
        [
            "--params",
            "env_odom_noise",
            "--coverage-dev-tol",
            "0.20",
            "--min-advance-rate",
            "0.70",
            "--max-false-advance-rate",
            "0.10",
            "--episodes-per-config",
            "30",
            "--seed",
            str(seed),
        ],
    ),
    (
        "20_multi_parameter",
        [
            "--params",
            "env_odom_noise,env_obs_noise,decay_factor,success_radius,gate_threshold",
            "--other-grid-n",
            "6",
            "--episodes-per-config",
            "20",
            "--seed",
            str(seed),
        ],
    ),
    (
        "21_max_confidence",
        [
            "--params",
            "env_odom_noise",
            "--env-odom-noise-range",
            "0.0",
            "0.15",
            "--other-grid-n",
            "12",
            "--sigma-visible-grid",
            "1e-4",
            "0.5",
            "20",
            "--odom-noise-grid",
            "1e-4",
            "0.3",
            "20",
            "--episodes-per-config",
            "200",
            "--seed",
            str(seed),
        ],
    ),
]


passed = []
failed = []

for i, (name, args) in enumerate(EXPERIMENTS, start=1):

    print("=" * 80)
    print(f"[{i}/{len(EXPERIMENTS)}] {name}")

    summary = RESULTS / f"s{seed}_{name}_summary.csv"
    details = RESULTS / f"s{seed}_{name}_details.csv"
    logfile = RESULTS / f"s{seed}_{name}.log"

    cmd = [
        PYTHON,
        "-u",
        SCRIPT,
        *args,
        "--out-summary",
        str(summary),
        "--out-details",
        str(details),
    ]

    start = time.time()

    with open(logfile, "w") as log:

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)

        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()
        proc.wait()

    elapsed = time.time() - start

    if proc.returncode == 0:
        print(f"[cool] Completed in {elapsed:.1f}s")
        passed.append(name)
    else:
        print(f"[bro] FAILED in {elapsed:.1f}s: err_{proc.returncode}")
        failed.append(name)

print("\n")
print("=" * 80)
print(f"Finished. in {calculate_time(time.time())}")
print(f"Passed : {len(passed)}")
print(f"Failed : {len(failed)}")

if failed:
    print("\nFailures:")
    for f in failed:
        print("  ", f)
