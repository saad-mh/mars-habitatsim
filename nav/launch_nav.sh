#!/usr/bin/env bash
# One-command bringup for the Mars habitat sim + actual-NavDP rover GUI.
#
#   ./nav/launch_nav.sh                       # empty yard, spawn at (0, 8)
#   ./nav/launch_nav.sh --rock-field rock_envs/run1/rock_field.json
#   ./nav/launch_nav.sh --start-x -2 --start-z 8 --start-yaw 120
#   ./nav/launch_nav.sh --no-cbf              # disable CBF obstacle avoidance
#
# In-house replacement for Nav_new/MARS/launch_mars.sh, built entirely under
# mars-habitatsim/nav/ -- no imports from Nav_new or its scripts/nav_pipeline.
# Drives with the real, published NavDP model (sam_vla.policy.navdp_upstream_policy,
# see rover_controller.py's docstring), not this repo's own in-house
# S2DiT+NavDP model.
#
# Unlike launch_mars.sh, there is no separate sim-node process and no Zenoh
# bridge: MarsHabitatEnv runs in-process (the `habitat` conda env already has
# everything nav/ imports -- see CLAUDE.md's env table), and the real NavDP
# model / the Qwen VLM used for one-shot goal resolution are spawned as
# subprocesses automatically by NavdpUpstreamPolicy / QwenServerManager, the
# same way sam_vla.run_navdp_rollout already does. So a single conda env and
# a single command is enough here.
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
# NOTE: no `set -u` -- some conda activate.d scripts in envs like this one
# chain into setup scripts that reference unset shell vars; tolerate that
# rather than failing the whole launch over it.
conda activate habitat

cd "$REPO"
echo "[nav] starting rover control GUI (habitat env, in-process sim + NavDP-upstream subprocess)..."
python -m nav.gui "$@"
