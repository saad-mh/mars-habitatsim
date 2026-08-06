# Two studies: human-in-the-loop uncertainty handoff, and real-world sensor-noise injection

## Status of the previous plan this file held

This file previously documented the belief-driven ghost-mask feature. That plan's Phase 0 (belief
growth + `sam_vla/core/ghost_mask.py`), Phase 1 (interactive verification in `kb_teleop_vl.py`), and
the ghost-mask-mode addendum (VLM-placed circle, `vl_direction`'s fourth mode) are all implemented and
merged. **Phase 2 (wiring the ghost mask into `sam_vla/run_navdp_rollout.py`'s real rollout loop) was
never done** — `grep -n "ghost_mask" sam_vla/run_navdp_rollout.py` currently returns nothing. If that's
still wanted, re-derive it from `git log -p -- next.md` (the plan text is intact in history) rather than
from memory — don't re-plan it from scratch. This file now tracks two new, unrelated studies.

---

# Study 1 — human-in-the-loop uncertainty handoff vs. autonomous driving (timing ablation)

## Research question

When the rover's goal-belief uncertainty crosses a threshold, does handing control to a human (stop,
rotate, let the human pick a heading, drive N units, repeat if still lost) reach the goal faster/more
reliably than letting the VLM/policy keep driving through the uncertainty on its own belief (mu +
covariance)? The ablation must isolate **pure driving/inference time** from **human decision time** and
**one-time model-load time**, so the two conditions are compared on a like-for-like clock.

## Existing building blocks (verified in this session)

| Piece | Where | What it gives us | What's missing |
|---|---|---|---|
| Uncertainty request/submit/retry protocol | [`vl_direction/uncertainty_session.py`](vl_direction/uncertainty_session.py) `UncertaintySession` | `request_human_heading()` → VLM sweep description (`NEEDS_HUMAN_INPUT`); `submit_heading(angle_deg\|angle_range_deg)` → `HEADING_DIRECTIVE` with `max_units`, no VLM call; `retry()` increments attempt and re-requests. | Purely a request/response object — nothing drives the rover along the returned heading, and nothing watches a real covariance to trigger it. |
| Real scalar uncertainty, already computed every step | [`sam_vla/core/belief_tracking.py`](sam_vla/core/belief_tracking.py) `BeliefGoalTracker.uncertainty_value()` | A real (not synthetic) per-step uncertainty state, grown via `uncertainty_growth_increment` while the goal is unseen, reset to `sigma_visible` on a fresh sighting — this **is** the "mu + uncertainty" state the VLM's autonomous condition should be conditioned on. Already live in `run_navdp_rollout.py`'s single-goal mode. | Nothing reads it against a threshold to trigger the uncertainty flow — `kb_teleop_vl.py` uses a separate synthetic proxy (`_update_uncertainty_covariance`), not this real tracker. |
| Session-mode flag | [`vl_direction/intervention/mode_flag.py`](vl_direction/intervention/mode_flag.py) `SessionMode.AUTONOMOUS`/`HUMAN_INTERVENED`, [`teleop_bridge.py`](vl_direction/intervention/teleop_bridge.py) | A ready-made tag for "this segment was human-driven" vs. "VLM-driven", read into every `VLDirectiveResult.session_mode`. | Not wired into the uncertainty flow anywhere — `kb_teleop_vl.py`'s halt never flips it, so today every uncertainty halt is still tagged `AUTONOMOUS`. |
| Aggregation shape | [`vl_direction/logging/hci_metrics.py`](vl_direction/logging/hci_metrics.py) `success_rate_by_mode`, `steps_to_goal_by_mode`, `uncertainty_trigger_stats` | Template for episode-level by-mode comparison. | Caller-supplied inputs only; no per-segment timing decomposition exists to feed it (see below). |
| One live reference integration | [`kb_teleop_vl.py`](kb_teleop_vl.py) `halted_for_uncertainty` flag, `_dispatch_uncertainty_request`/`_uncertainty_worker`/`_handle_halted_key` (background-thread VLM call + `root.after` marshal back to Tk, numpad heading keys) | Working halt/resume UI pattern to model a rollout-loop version on. | It's a **manual demo**: after `submit_heading`, the human just resumes WASD control for the announced `max_units` — there is no autonomous "drive N units along heading" code anywhere in the repo, and no timestamp brackets the halt at all. |
| Per-call VLM latency | [`vl_direction/directive_engine.py`](vl_direction/directive_engine.py) `query()` (`t0`/`latency_ms`, L184/196) | Already isolates one VLM call's wall time into `VLDirectiveResult.latency_ms` — this is directly usable as the "VLM inference time" bucket for the request-phase call. `submit_heading` makes no VLM call (near-zero, packaging only). | Not threaded into any per-step log — neither `run_navdp_rollout.py` nor `run_qwen_vla_rollout.py` call `directive_engine.query` at all today (only `exploration_policy.py` does). |
| Model load/startup | [`sam_vla/vlm/qwen_server_manager.py`](sam_vla/vlm/qwen_server_manager.py) `QwenServerManager.start()`, `vl_direction/internvl_server_manager.py` `InternVLServerManager.start()` | Both already block on a health-check poll loop until the subprocess model server is up. | The elapsed duration is never captured — `start()` has no `t0`/`t1` at all, just a deadline to give up by. |
| Motion primitive | [`sam_vla/core/pose_integrator.py`](sam_vla/core/pose_integrator.py) `integrate_mars(pose, action, dt)`, called at [`run_navdp_rollout.py:713-714`](sam_vla/run_navdp_rollout.py:713) | Exact kinematic step used for every real rollout action — the same primitive an autonomous "drive toward heading_deg" loop would call repeatedly with `v_fwd` fixed and `yaw_rate` steering toward the target heading. | No existing helper turns "heading_deg + max_units" into a sequence of such calls; would need a small new loop. |
| Goal-visibility predicate | Already used at the (unimplemented) ghost-mask Phase 2 call site, `not goal_visible` | The exact condition for "goal still not spotted after N units, retry" | N/A — just reuse it. |

## What "the VLM keeps driving" (control condition) means here

Re-reading the ask: Condition B is not "give the VLM an autonomous heading-picking flow instead of the
human" — it's simpler and needs no new heading-selection logic: **the normal rollout loop (NavDP/Qwen
policy + CBF) just keeps running unmodified**, conditioned on `BeliefGoalTracker`'s own mu/uncertainty
as it already is today, and the uncertainty-crossing event is only *logged*, never acted on. Condition A
is the same rollout loop with the stop-rotate-ask-drive-retry cycle spliced in at the same trigger
points. This keeps the two conditions paired on identical trigger instances (same seed/episode/step),
which is what makes the timing subtraction meaningful.

## Timing decomposition (the actual deliverable of this ablation)

Four buckets, each needing an explicit timer added at a specific site — none of these exist today:

1. **`model_load_ms`** — one-time per process. Add `t0 = time.monotonic()` before, `t1` after, in
   `QwenServerManager.start()` / `InternVLServerManager.start()`; log once per episode (or once per
   process if the server is reused across episodes — note which in the log so it isn't double-counted).
2. **`vlm_inference_ms`** — already exists as `VLDirectiveResult.latency_ms` on the `request_human_heading()`
   call; just needs to be threaded through to the per-event log record instead of discarded.
3. **`human_decision_ms`** — new. Timestamp when the sweep description / frame is presented to the human
   (start of the blocking/async wait for input) to when their heading choice is received (right before
   `submit_heading()` is called). This is the field that gets *deducted* from Condition A's total time to
   recover "VLM/driving-only" time for a fair comparison against Condition B.
4. **`drive_ms`** — new. Wall time of the autonomous "drive up to `max_units` along `heading_deg`" loop
   (Phase 1 below), timestamped independently of steps spent in normal policy-driven navigation.

Per-episode ablation arithmetic: `total_episode_time - sum(human_decision_ms) - model_load_ms` gives
Condition A's VLM-and-driving-only time, directly comparable to Condition B's total time (which has no
human-decision or extra model-load component beyond the one shared model already running).

## Phased plan

### Phase 0 — real trigger + logging scaffolding (no behavior change yet) — DONE

- `--uncertainty-threshold` CLI flag wired in `sam_vla/run_navdp_rollout.py`'s single-goal path
  ([run_navdp_rollout.py:1113-1121](sam_vla/run_navdp_rollout.py:1113)), checked every step against
  `BeliefGoalTracker.uncertainty_value()` ([run_navdp_rollout.py:724-765](sam_vla/run_navdp_rollout.py:724)).
  Rising-edge only (`uncertainty_was_above` gate) so a long occlusion logs one `uncertainty_trigger`
  event, not one per step spent above threshold. Currently always tags `condition: "autonomous"` with
  every timing bucket `None` — Phase 2/3 is what actually populates them.
- `RolloutLogger.log_step` takes an optional `uncertainty_event: Optional[dict]`
  ([rollout_logger.py:49-65](sam_vla/logging/rollout_logger.py:49)), written into each step's manifest
  entry (`None` on every step that isn't a trigger).
- `model_load_ms` timer hooks are live in both server managers' `start()` —
  [qwen_server_manager.py:77-104](sam_vla/vlm/qwen_server_manager.py:77) and
  [internvl_server_manager.py:89-129](vl_direction/internvl_server_manager.py:89) — `0.0` if a server
  was already running (nothing to attribute to this process's load), else wall time from `t0` to the
  first successful health check.

### Phase 1 — autonomous motion primitives (shared by both conditions' plumbing) — DONE

New module [`sam_vla/core/uncertainty_motion.py`](sam_vla/core/uncertainty_motion.py), pure/sim-facing
(env is duck-typed: `step(pose)`, `get_observation(frame_idx)`, `get_full_observation(frame_idx)` — no
VLM/UncertaintySession import), unit-tested against a fake env in
[`sam_vla/core/tests/test_uncertainty_motion.py`](sam_vla/core/tests/test_uncertainty_motion.py) (11
tests, run via `/home/gpu/miniconda3/envs/habitat/bin/python -m pytest
sam_vla/core/tests/test_uncertainty_motion.py`):

- `yaw_rate_toward_heading(current_yaw, target_yaw, turn_kp, max_yaw_rate) -> float` — the steering law
  itself, pulled out so it's testable with no env at all; wraps the heading error to `(-pi, pi]` first so
  it always turns the short way.
- `rotate_sweep(env, pose, degrees_per_step, n_steps, dt=0.1) -> list[(pose, rgb)]` — in-place rotation
  via `integrate_mars` + `env.step`, capturing a frame per step for
  `request_human_heading(frame)`/`retry(new_frame)`. No translation, so `(x, z)` is unchanged when it
  returns.
- `drive_toward_heading(env, pose, heading_deg, max_units, goal_visible_fn, v_fwd=0.5, turn_kp=1.4,
  max_yaw_rate=1.0, dt=0.1) -> (final_pose, units_covered, found)` — repeatedly calls
  `integrate_mars`/`env.step` with `v_fwd` fixed, steering via `yaw_rate_toward_heading`, stopping early
  once `goal_visible_fn(obs)` (obs = `env.get_full_observation()`, so callers can reuse the same
  `MESH_GOAL_ID`-mask check the main rollout loop uses) is true or `max_units` m is covered. This is the
  piece that didn't exist anywhere in the repo before (confirmed: `kb_teleop_vl.py`, now at
  `scripts/habitat_tests/kb_teleop_vl.py`, only ever resumes manual WASD after `submit_heading`).

  **Important convention note for Phase 2**: `heading_deg` here is an ABSOLUTE world heading (`Pose.yaw`'s
  convention, degrees), *not* the rover-front-relative angle `UncertaintySession.submit_heading` deals in
  (see `kb_teleop_vl.py`'s numpad mapping, `UNCERTAINTY_HEADING_KEYS` — 8=0°/front, 6=+90°/right,
  4=-90°/left, positive=clockwise=right). Phase 2 must convert: capture the world yaw *before*
  `rotate_sweep` spins the rover around (that's the human's "front" reference), then
  `target_world_yaw_deg = reference_yaw_deg - human_angle_deg` (positive human angle = turn right = yaw
  decreases, since `Pose.yaw` is CCW-positive) before calling `drive_toward_heading`.

### Phase 2 — Condition A (human-in-the-loop) wiring — DONE

Resolved open question: **Tk popup**, not a terminal prompt (user's explicit choice — trades headless/
SSH-friendliness for showing the human the actual sweep frame). Implemented directly in
[`sam_vla/run_navdp_rollout.py`](sam_vla/run_navdp_rollout.py) rather than a new module, since Study 1 is
specific to this entry point:

- `_prompt_human_heading(frame_rgb, sweep_description, attempt, max_retries) -> angle_deg` — modal Tk
  popup (blocks on `mainloop()`), numpad-style buttons mirroring `kb_teleop_vl.py`'s
  `UNCERTAINTY_HEADING_KEYS` layout (`_UNCERTAINTY_HEADING_LAYOUT`), showing the sweep frame + the VLM's
  sweep description. Falls back to `0.0` (front) if the window is closed without a pick, rather than
  raising. Needs a real display — that's the accepted tradeoff of the Tk choice.
- `_run_uncertainty_handoff(env, pose, step, episode_id, client, current_uncertainty,
  uncertainty_threshold, dt, sweep_degrees_per_step, sweep_steps, max_units, max_retries, drive_v_fwd,
  lost_goal_min_px, prompt_fn=None) -> dict` — the actual stop-rotate-ask-drive-retry loop: `rotate_sweep`
  → `UncertaintySession.request_human_heading`/`.retry` (`vlm_inference_ms` accumulated from
  `result.latency_ms`) → `prompt_fn` (timed into `human_decision_ms`) → `submit_heading` →
  `resolve_absolute_heading_deg` (converts the human's rover-front-relative angle into the absolute world
  heading `drive_toward_heading` needs, using the yaw captured *before* `rotate_sweep` ran as the "front"
  reference) → `drive_toward_heading` (timed into `drive_ms`), looping up to `max_retries` if not
  resolved. Flips `SessionMode` to `HUMAN_INTERVENED` for the duration via a `try/finally` (guaranteed
  reset even if `prompt_fn` raises — tested). `prompt_fn` is injectable so this whole loop is unit-tested
  (4 tests in
  [`sam_vla/tests/test_run_navdp_rollout_uncertainty.py`](sam_vla/tests/test_run_navdp_rollout_uncertainty.py))
  against a fake env + `MockInternVLClient`, with no real display, sim, or live VLM.
- A **fresh `UncertaintySession` is constructed per trigger** (not reused across triggers/episode), with
  `covariance_value=current_uncertainty` — `UncertaintySession.covariance_value` is fixed at
  construction, so this is the only way to have each trigger's VLM context reflect the *actual* live
  uncertainty without changing `vl_direction`'s own contract (the stated non-goal).
- **Backend note**: uses `get_client("qwen")` / `vl_direction.qwen_server_manager.QwenServerManager`
  (confirmed real and working, same as `kb_teleop_vl.py`'s live integration) rather than the InternVL
  backend/`InternVLServerManager` the "Existing building blocks" table above names — InternVL's model
  runner is still `NotImplementedError` stubs (see `vl_direction/client.py`'s module docstring), so it
  can't actually serve a request yet. Added the same `model_load_ms` timer hook to
  `vl_direction/qwen_server_manager.py` that Phase 0 added to the other two managers (it hadn't been
  covered, since Phase 0's table didn't know Phase 2 would end up needing this specific one). New CLI
  flags: `--uncertainty-condition {autonomous,human}` (default `autonomous`, i.e. unchanged Condition B
  behavior), `--uncertainty-max-retries`, `--uncertainty-max-units`,
  `--uncertainty-sweep-degrees-per-step`, `--uncertainty-sweep-steps`, `--uncertainty-drive-v-fwd`,
  `--uncertainty-mock-client` (swaps in `MockInternVLClient`, skipping the real server spawn — for
  exercising the flow without a live `qwen_vlm` server).
- `model_load_ms` is charged once per episode (first trigger only; `0.0` on subsequent triggers in the
  same run) via a `uncertainty_model_load_charged` flag — resolves Phase 0's open question in favor of
  "once per process/episode," since each `run()` call is one episode with its own server subprocess.

**Not yet exercised end-to-end against the real sim** (no GPU/display run performed as part of this
change — only unit/orchestration tests above): a real `--uncertainty-condition human` rollout still needs
a live `qwen_vlm` server, a display for the Tk popup, and a real `--ckpt`/`--scene-path` to actually try.

### Phase 3 — Condition B (autonomous baseline) + paired runs

- Same trigger, but the handler is a no-op besides logging `uncertainty_trigger` (per "what Condition B
  means" above) — policy keeps acting on its own belief-conditioned state uninterrupted.
- Run both conditions over the same seed/episode set (same start pose, same goal, same scene) so trigger
  instances line up 1:1 for the subtraction described above.

### Phase 4 — analysis

- Per-episode: `time_to_goal`, `success`, `n_uncertainty_triggers`, `sum(human_decision_ms)`,
  `sum(drive_ms)`, `model_load_ms`, derived `vlm_and_driving_only_time`.
- Compare Condition A's derived time and success rate against Condition B's raw time and success rate —
  this is the actual ablation result (does human intervention win once its own decision latency is
  factored out, or does it still lose because rotate+ask+drive is fundamentally slower per trigger?).

## Non-goals

- Don't change `vl_direction`'s public contract (`UncertaintySession`, `query()`, schemas) — extend the
  *callers*, not the module (same discipline the ghost-mask plan used for `vl_direction`).
- Don't touch `sam_vla/policy/base_policy.py`'s `NavigationPolicy` protocol or `CbfObstacleAvoidance.apply()`
  — the uncertainty handoff sits *around* a policy step, not inside one.
- Don't conflate this with the ghost-mask feature — that's a passive visualization cue; this is an active
  control handoff. Keep them separate even though both key off the same `BeliefGoalTracker` uncertainty
  state.

## Open questions

- Terminal prompt vs. a small Tk popup for the human-input surface in Phase 2 — terminal is simpler and
  keeps the rollout scripts headless-capable, but loses the visual sweep frames a human would actually
  want to see before choosing. Decide at implementation time based on whether these runs need to happen
  unattended/batched (favors terminal + pre-recorded frame save) or supervised (favors Tk).
- Exact uncertainty threshold and `max_units` per attempt — reuse whatever the ghost-mask work's pixel/
  uncertainty clamp constants suggest as a starting point, tune empirically.
- Whether `model_load_ms` should be amortized across all episodes in a batch run (server started once,
  reused) or charged once per episode (server restarted per episode for clean isolation) — affects how
  directly Condition A/B totals are comparable if run in the same process vs. separate processes.

---

# Study 2 — real-world sensor/actuator noise injected into driving deltas

## Research question

`belief_exp` established (offline, numpy-only) which odometry-noise magnitudes the belief system stays
well-calibrated under. This study applies that same noise magnitude to the **actual** driving deltas in
a real Habitat-Sim rollout — not a numpy approximation — and observes what really happens to task
performance (success rate, time/steps to goal, CBF-trigger rate) and to the belief tracker's own
uncertainty trajectory when real noise, not zero, drives the rover.

## Important caveat: `belief_exp/results/` is currently empty on disk

`ls belief_exp/results/` returns nothing in the working tree — the CSVs were deliberately deleted in
commit `63d7869` ("Delete obsolete sigma_min_summary CSV files"). The numbers below were recovered via
`git show 63d7869^:belief_exp/results/<file>`, from `belief_exp/sigma_min_sweep.py`'s
`21_max_confidence_summary.csv` (the widest sweep, 200 episodes/config):

| `env_odom_noise_std` | min viable `sigma_visible` | min viable `odom_noise` (bank) | viable? |
|---|---|---|---|
| 0.0 | 0.0089 | 0.0001 (grid floor) | yes |
| 0.041 | 0.0139 | 0.0001 | yes |
| 0.068 | 0.0217 | 0.0001 | yes |
| 0.109 | 0.0217 | 0.0001 | yes |
| 0.15 (grid max) | 0.0340 | 0.0001 | yes |

An earlier, stricter run (`07_strict_viability_summary.csv`) went fully non-viable above
`env_odom_noise_std ≈ 0.075`. **Before trusting any specific number for this study, either re-run
`belief_exp/sigma_min_sweep.py`/`sweep.py` fresh, or pull the exact historical file via `git show`** —
don't treat the table above as a standing artifact, only as a recovered snapshot.

Two different things are called "sigma" in `belief_exp` and it matters which one this study injects:
- **`env_odom_noise_std`** (scenario-side): the actual/ground-truth noise magnitude applied to real
  motion deltas inside `belief_exp/scenario.py` — this is the one that's the analog of real-world
  sensor/actuator noise, and the one this study should port into the real rollout's driving deltas.
- **`sigma_visible`/`odom_noise` (bank-side)**: `SubgoalBeliefBank`'s own *belief* about how noisy things
  are — a calibration/tuning parameter, not a source of physical noise. Leave these alone; they're
  already exposed via the existing `--belief-odom-noise` flag and only affect the belief tracker's
  internal dead-reckoning, not real motion (confirmed: `sam_vla/core/belief_tracking.py`'s
  `propagate_body_point` perturbs a local copy used only for `belief_g`, never the `action` object
  actually passed to `integrate_mars`).

So: **sweep `env_odom_noise_std` over the range `belief_exp` already tested (0.0 to ~0.15, extending to
the ~0.075 non-viable boundary as a stress point) and inject it into real driving deltas**, using the
`min_sigma_visible`/bank-config table above only as *context* for what the belief tracker's calibration
would need to look like to stay honest at each noise level — not as the thing being injected.

## Exact noise formula to port (from `belief_exp/scenario.py:144-151`)

```python
odom_noise_xy = env_odom_noise_std
odom_noise_th = env_odom_noise_std * 0.5
noisy_dx    = true_dx    + rng.normal(0.0, odom_noise_xy)
noisy_dy    = true_dy    + rng.normal(0.0, odom_noise_xy)
noisy_dtheta = true_dtheta + rng.normal(0.0, odom_noise_th)
```
Full-scale Gaussian noise on translational deltas, half-scale on the heading delta. Per the existing
`belief_exp`/`navdp` convention ("never copies or reimplements the real classes"), and the equivalent
precedent from the ghost-mask plan ("port the *pattern*, not the import" — `ghost_mask.py` vs.
`navdp/navdp/extensions/ghost_geometry.py`), **port this formula into a new `sam_vla/` module rather than
importing `belief_exp/scenario.py`** — `belief_exp` stays a read-only offline harness, untouched.

## Injection point

Confirmed exact site: [`sam_vla/run_navdp_rollout.py:713-714`](sam_vla/run_navdp_rollout.py:713):
```python
new_pose = integrate_mars(obs.pose, action, dt)
env.step(new_pose)
```
`integrate_mars` ([`sam_vla/core/pose_integrator.py:18-27`](sam_vla/core/pose_integrator.py:18)) is pure
SE(2) kinematics; `MarsHabitatEnv.step()` just teleports to the given absolute pose — no noise anywhere
in this path today. Perturb `action.v_fwd`/`action.v_lat`/`action.yaw_rate` (equivalently, perturb the
`dx`/`dz`/`dtheta` `integrate_mars` computes internally) with the ported formula immediately before this
call.

## New building block: `sam_vla/core/sensor_noise.py`

- `apply_odom_noise(action: Action, odom_noise_std: float, rng: np.random.Generator) -> Action` — pure
  function, returns a new `Action` with the scenario.py formula applied (full-scale on `v_fwd`/`v_lat`,
  half-scale on `yaw_rate`, matching the dx/dy/dtheta split since these are the same physical quantities
  pre-integration). Unit-testable with no sim dependency, same style as Phase 0's `ghost_mask.py` pure
  functions.
- Called at `run_navdp_rollout.py:713`, gated by a new `--drive-odom-noise-std` CLI flag (distinct name
  from the existing `--belief-odom-noise`, to keep "real motion noise" and "belief's own noise-belief"
  unambiguous in logs and flag help text).

## Sweep design (this time actually running the simulation)

- Reuse an existing checkpoint/scene/goal config (whatever the current default rollout invocation in
  `usage` uses) so this is a controlled variable, not a confound.
- For each `env_odom_noise_std` in the recovered/re-derived range (e.g. `{0.0, 0.041, 0.068, 0.109, 0.15,
  ~0.075-stress-point}`), run N real episodes (N chosen for statistical power — `belief_exp`'s own sweeps
  used 60-200 episodes/config as a reference point, though real Habitat rollouts are far more expensive
  per-episode so N will likely be much smaller; decide based on available compute).
- Per episode, log (via `RolloutLogger`/`EpisodeLogger`, extending as needed): success, steps/time to
  goal, final distance to goal, CBF-trigger count (existing `distances_to_obstacles`/`cbf_active` fields
  already support this), and the belief tracker's own `uncertainty_value()` trajectory for direct
  comparison against `belief_exp`'s offline `calibration_nll`/`coverage_1sigma`/`coverage_2sigma` at the
  same nominal noise level — this is the actual "does the offline numpy approximation hold up in the real
  sim" validity check the study is going after.

## Non-goals

- Don't modify anything under `navdp/` or `belief_exp/` — this study is a new consumer of `belief_exp`'s
  *findings* (the noise range), not a modification of how those findings were produced.
- Don't feed this noise into `BeliefGoalTracker.odom_noise`/`propagate_body_point` — that parameter
  already exists and is deliberately orthogonal (belief's internal noise-belief vs. real physical noise);
  conflating them would break the exact distinction `belief_exp` was built to study.
- Don't change `NavdpPolicy`/`QwenDiscreteDirectionPolicy` — noise is injected after the policy decides
  an action, not into the policy's inputs.

## Open questions

- Exact episode count per noise level, given real Habitat rollouts are much more expensive than
  `belief_exp`'s numpy episodes — depends on available compute/time budget, not resolvable in the
  abstract.
- Whether to re-run `belief_exp/sigma_min_sweep.py` fresh (to get results actually present on disk and
  reproducible going forward) before starting Study 2, vs. just citing the recovered historical numbers
  above with a note — recommend re-running, since the table above is explicitly flagged as
  git-history-only and not a standing artifact.
- Whether `--drive-odom-noise-std` should also perturb the *initial* `env.step()` teleport target
  directly (pose-space noise) instead of/in addition to the pre-integration action-space noise above —
  the scenario.py formula is action/delta-space, so action-space is the faithful port, but worth
  confirming there's no meaningful difference given `integrate_mars` is a simple linear transform.
