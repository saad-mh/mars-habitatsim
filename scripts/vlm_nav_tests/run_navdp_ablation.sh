#!/usr/bin/env bash
set -euo pipefail

# Edit these paths once, or override any of them as environment variables.
HABITAT_PYTHON="${HABITAT_PYTHON:-/home/gpu/miniconda3/envs/habitat/bin/python}"
MARS_ROOT="${MARS_ROOT:-/home/gpu/Desktop/pineapple/mars-habitatsim}"
NAVDP_ROOT="${NAVDP_ROOT:-/home/gpu/Desktop/pineapple/navdp_upstream}"
CHECKPOINT="${CHECKPOINT:-${MARS_ROOT}/navdp/navdp-cross-modal.ckpt}"
SCENE="${SCENE:-${MARS_ROOT}/assets/marsyard2022.glb}"
TERRAIN_OBJ="${TERRAIN_OBJ:-${MARS_ROOT}/assets/marsyard2022.obj}"
ROLLOUT_SCRIPT="${ROLLOUT_SCRIPT:-${MARS_ROOT}/scripts/vlm_nav_tests/rollout_s2dn_policy.py}"
ANALYZER_SCRIPT="${ANALYZER_SCRIPT:-${MARS_ROOT}/scripts/vlm_nav_tests/analyze_navdp_ablation.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MARS_ROOT}/runs/navdp_ablation}"

# Space-separated seeds. Use at least 30 seeds for paper results.
SEEDS="${SEEDS:-7 8 9}"
RUN_COMPONENTS="${RUN_COMPONENTS:-0}"

START_X="${START_X:-8}"
START_Z="${START_Z:-10}"
START_YAW_DEG="${START_YAW_DEG:-0}"
GOAL_X="${GOAL_X:-8}"
GOAL_Z="${GOAL_Z:--8}"
OBSTACLE_UV="${OBSTACLE_UV:-0.50,0.68}"

for required_file in \
    "${HABITAT_PYTHON}" \
    "${CHECKPOINT}" \
    "${SCENE}" \
    "${TERRAIN_OBJ}" \
    "${ROLLOUT_SCRIPT}" \
    "${ANALYZER_SCRIPT}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done

COMMON=(
    --navdp-root "${NAVDP_ROOT}"
    --navdp-checkpoint "${CHECKPOINT}"
    --navdp-python "${HABITAT_PYTHON}"
    --navdp-device cuda:0
    --scene "${SCENE}"
    --terrain-obj "${TERRAIN_OBJ}"
    --start-x "${START_X}"
    --start-z "${START_Z}"
    --start-yaw-deg "${START_YAW_DEG}"
    --goal-x "${GOAL_X}"
    --goal-z "${GOAL_Z}"
    --obstacle-mode mesh
    --obstacle-mesh-uv "${OBSTACLE_UV}"
    --mesh-half-pixels 32
    --maximum-obstacle-depth 12.0
    --safe-distance 1.20
    --hard-collision-distance 0.35
    --safety-weight 80
    --guidance-strength 1.0
    --temperature 0.25
    --particle-std 0.28
    --candidates 24
    --particles 12
)

HLC=(
    --barrier-weight 30
    --barrier-rate 0.15
    --circulation-weight 25
    --circulation-activation-distance 2.0
    --circulation-activation-sharpness 0.25
    --minimum-circulation-progress 0.035
    --blocking-alignment-threshold 0.15
    --circulation-switch-weight 3.0
    --escape-lateral-target 0.40
)

BASE_S2DIFF=(
    --barrier-weight 0
    --circulation-weight 0
    --circulation-switch-weight 0
    --escape-lateral-target 0
)

run_episode() {
    local label="$1"
    local seed="$2"
    shift 2
    local output="${OUTPUT_ROOT}/${label}/seed_${seed}"
    echo
    echo "=== ${label}, seed=${seed} ==="
    "${HABITAT_PYTHON}" "${ROLLOUT_SCRIPT}" \
        "${COMMON[@]}" \
        --seed "${seed}" \
        --output "${output}" \
        "$@"
}

for seed in ${SEEDS}; do
    # Released NavDP denoising and learned critic selection.
    run_episode pure_navdp "${seed}" \
        --planner-mode pure-navdp

    # S2Diff energy without the proposed barrier/circulation/latch/escape package.
    run_episode s2diff_base "${seed}" \
        --planner-mode s2diff \
        --remove-critic \
        "${BASE_S2DIFF[@]}"

    # Full method, but retain the unused critic head as the critic-removal control.
    run_episode hlc_keep_critic "${seed}" \
        --planner-mode s2diff \
        --no-remove-critic \
        "${HLC[@]}"

    # Full proposed model.
    run_episode hlc_no_critic "${seed}" \
        --planner-mode s2diff \
        --remove-critic \
        "${HLC[@]}"

    if [[ "${RUN_COMPONENTS}" == "1" ]]; then
        run_episode barrier_only "${seed}" \
            --planner-mode s2diff --remove-critic \
            --barrier-weight 30 \
            --circulation-weight 0 \
            --circulation-switch-weight 0 \
            --escape-lateral-target 0

        run_episode circulation_only "${seed}" \
            --planner-mode s2diff --remove-critic \
            --barrier-weight 0 \
            --circulation-weight 25 \
            --circulation-activation-distance 2.0 \
            --minimum-circulation-progress 0.035 \
            --circulation-switch-weight 0 \
            --escape-lateral-target 0

        run_episode hlc_no_latch "${seed}" \
            --planner-mode s2diff --remove-critic \
            "${HLC[@]}" \
            --circulation-switch-weight 0

        run_episode hlc_no_escape "${seed}" \
            --planner-mode s2diff --remove-critic \
            "${HLC[@]}" \
            --escape-lateral-target 0
    fi
done

ANALYZER_ARGS=()
LABELS=(pure_navdp s2diff_base hlc_keep_critic hlc_no_critic)
if [[ "${RUN_COMPONENTS}" == "1" ]]; then
    LABELS+=(barrier_only circulation_only hlc_no_latch hlc_no_escape)
fi

for label in "${LABELS[@]}"; do
    for seed in ${SEEDS}; do
        ANALYZER_ARGS+=(
            --run "${label}=${OUTPUT_ROOT}/${label}/seed_${seed}/rollout.npz"
        )
    done
done

"${HABITAT_PYTHON}" "${ANALYZER_SCRIPT}" \
    "${ANALYZER_ARGS[@]}" \
    --output "${OUTPUT_ROOT}/comparison"

echo
echo "Finished. Summary:"
echo "  ${OUTPUT_ROOT}/comparison.csv"
echo "  ${OUTPUT_ROOT}/comparison.json"
