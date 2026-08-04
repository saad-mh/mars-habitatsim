# Belief-driven ghost mask (VLM exploration cue)

## Original ask (verbatim intent, for reference)

1. Don't plug belief into NavDP/any policy directly. Instead plug it into a VLM/LM/Model,
   which draws a translucent "ghost mask" — a circular patch, radius optionally proportional
   to uncertainty — on the frame, given the belief, when the task is to explore. Non-destructive
   to the underlying render; semi-transparent green.
2. The policy stays belief-blind; belief only ever reaches it indirectly, through the VLM.
   Clear separation of concerns between policy and belief-visualization.
3. The ghost mask updates in real time as belief changes (grows/shrinks/moves as the agent
   gathers more information).
4. The policy keeps operating on its own internal state/decisions, taking the ghost mask as a
   directional/visual cue — i.e. it's supposed to influence driving, eventually.

## What "ghost mask" and "belief" resolve to in this codebase

Two readings were possible for "belief when the task is to explore." Given what already exists
(below), the coherent one is: **this is the existing "lost-goal" ghost** — the goal is known but
currently out of view, dead-reckoned by [`BeliefGoalTracker`](sam_vla/core/belief_tracking.py),
and "explore" means "normal driving, not currently doing CBF obstacle-avoidance" (`vl_direction`
already models exactly this binary: `"cbf"` vs `"exploration"` mode, see
[kb_teleop_vl.py:328-333](kb_teleop_vl.py:328)). It is *not* a frontier/unknown-space belief —
there's no such representation anywhere in this repo.

## Confirmed decisions (from planning discussion)

- **Target surface, phased**: build the belief-growth + ghost-drawing logic as one shared,
  policy-agnostic piece first; verify it interactively in
  [kb_teleop_vl.py](kb_teleop_vl.py); then wire the same piece into the real rollout loop in
  [sam_vla/run_navdp_rollout.py](sam_vla/run_navdp_rollout.py) as a follow-up phase. Do not
  attempt both surfaces in one pass.
- **Uncertainty source**: extend [`BeliefGoalTracker`](sam_vla/core/belief_tracking.py) with a
  scalar `uncertainty` state, numpy-only (no `navdp.extensions`/torch dependency — keeps it
  runnable in the `habitat` conda env that `kb_teleop_vl.py` and the sam_vla rollout scripts
  already use). Growth is driven by **two configurable flags**, not a hardcoded constant:
  - `odom_noise` — already an existing constructor param (currently only perturbs the
    dead-reckoned point in `propagate_body_point`); reuse it as the per-step base growth of
    `uncertainty` while the goal is unseen.
  - a new `odom_noise_growth_rate` — how much faster `uncertainty` accumulates the longer the
    goal stays unseen (an accelerating-drift term, e.g. `increment = odom_noise * (1 +
    odom_noise_growth_rate * time_since_seen)` — exact formula is an implementation-time choice,
    see Open Questions).
  - On a fresh sighting (`observe()` returns `True`), reset `uncertainty` to a small floor
    (mirrors `SubgoalBeliefBank`'s `sigma_visible` semantics in
    [navdp/navdp/extensions/belief_bank.py](navdp/navdp/extensions/belief_bank.py) conceptually,
    without importing it).
  - The real `navdp.extensions.SubgoalBeliefBank` (real Kalman `Sigma`) stays available as a
    heavier alternative later if the numpy approximation proves insufficient, but is explicitly
    out of scope for this pass — it only runs in the `sam2`/`sam3` conda envs, not `habitat`.
- **Action coupling**: **advisory-only**. The VLM's returned `Direction` (LEFT/RIGHT/FRONT/BACK)
  is rendered on-screen and logged, exactly like `kb_teleop_vl.py` already does for its
  cbf/exploration calls today (`_vl_worker` only ever prints `result.direction`, never touches
  an `Action`). No `Action`, `NavigationPolicy`, or `CbfObstacleAvoidance` code changes in this
  pass. Point 4's "influence driving" is deferred to a later, separately-scoped change once the
  visualization itself is validated.

## Existing building blocks to reuse (verified in this session)

| Piece | Where | What it gives us | What's missing for this feature |
|---|---|---|---|
| Ghost-circle projection math | [navdp/navdp/extensions/ghost_geometry.py](navdp/navdp/extensions/ghost_geometry.py) (`gc_intrinsics`, `gc_body_point`, `gc_project`, `gc_make_mask`) | Pure numpy: body-frame bearing/range → world point → pixel (u,v) → filled-circle boolean mask. No torch dependency. | Lives under `navdp/`, which is treated as read-only/external (per CLAUDE.md); radius is a fixed constant, not uncertainty-driven. **Port the math pattern into `sam_vla/`, don't import from `navdp/`.** |
| Translucent alpha-blend | [sam_vla/perception/semantic_overlay.py:17-52](sam_vla/perception/semantic_overlay.py) `overlay_semantic_masks()` | Established green=goal / red=obstacle convention, `alpha=0.45` default, already used to prep frames for a VLM call (`qwen_discrete_direction_policy.py`, `run_navdp_rollout.py`). | Operates on a semantic-id frame + fixed `goal_id`, not a freestanding circle mask. |
| Circular translucent overlay (closest template) | [kb_teleop_vl.py:141-155](kb_teleop_vl.py:141) `overlay_obstacles()` | Vectorized `(x-cx)^2+(y-cy)^2<=r^2` mask + alpha blend, already circular, already non-destructive (returns a copy). Same file we're wiring Phase 1 into. | Hardcoded red; radius list comes from known obstacle geometry, not belief uncertainty. |
| Existing per-step belief tracker | [sam_vla/core/belief_tracking.py](sam_vla/core/belief_tracking.py) `BeliefGoalTracker` | `.observe(goal_mask, depth)`, `.propagate(action, dt)`, `.bearing()`, `.distance()` — already the "lost-goal" belief used in the real rollout loop. | No uncertainty/covariance field at all today — this is the actual gap to fill. |
| VLM exploration dispatch | [vl_direction/directive_engine.py](vl_direction/directive_engine.py) `query("exploration", frames, ExplorationContext(...), episode_id)` | Already takes `list[np.ndarray]` frames and returns a parsed `Direction`; contract is stable and shouldn't change. | Nothing — reuse as-is. Ghost mask is drawn *before* this call, into the frame(s) passed in. |
| Real rollout call sites (Phase 2 target) | [sam_vla/run_navdp_rollout.py:457-465](sam_vla/run_navdp_rollout.py:457) (`belief_tracker = BeliefGoalTracker(...)`, single-goal mode only), `:668-710` (`.observe`/`.bearing`/`.propagate` per step), `:679-685` (`blocked = avoidance.is_blocked(...)`, `lost_goal_ghost` gating), `:728-730` (`vis_rgb = overlay_semantic_masks(obs.rgb, semantic_render, text=...)`) | A live `belief_tracker` instance already exists per-step with bearing/distance available, plus an existing "not blocked" gate and an existing `vis_rgb` overlay call site that is *already* visualization-only (never fed back into the policy). | Nothing structural — Phase 2 is additive at these exact lines. |

## Explicit non-goals / anti-patterns to avoid

- **Do not** modify anything under `navdp/` (belief_exp's existing rule — "never copies or
  reimplements" — extends here too: port the *pattern*, not the import).
- **Do not** repeat `navdp/scripts/rollout_habitat_policy.py:983-1004`'s existing ghost pattern,
  which overwrites the policy's `goal_channel` input directly with the ghost circle. That's
  belief reaching the policy *directly* — exactly what point 2 of the original ask rules out.
  This plan's ghost mask only ever touches a `vis_rgb`/frame-for-VLM copy, never the tensor the
  policy consumes.
- **Do not** change `sam_vla/policy/base_policy.py`'s `NavigationPolicy` protocol, `Action`, or
  `sam_vla/safety/cbf_avoidance.py`'s `apply()` in this pass (advisory-only decision above).
- **Do not** change `vl_direction`'s public contract (`query()`, `schemas.py`,
  `ExplorationContext`) — it already accepts arbitrary annotated frames; reuse it unmodified.

## Phased plan

### Phase 0 — shared belief-growth + ghost-drawing module

New/modified files:
- **Modify** [sam_vla/core/belief_tracking.py](sam_vla/core/belief_tracking.py):
  add `uncertainty: float` state to `BeliefGoalTracker`, a `sigma_visible` floor constant, the
  new `odom_noise_growth_rate` constructor param, growth logic in `propagate()`, reset logic in
  `observe()`, and an accessor (`.uncertainty_value()` or similar).
- **New** `sam_vla/core/ghost_mask.py` (name tentative): pure-numpy, mirrors
  `ghost_geometry.py`'s shape but built against `sam_vla.core.goal_geometry.intrinsics_from_hfov`
  (already imported by `belief_tracking.py`) and the body-frame `[forward, left]` convention
  `BeliefGoalTracker` already uses, so it composes directly with `.bearing()`/`.distance()`
  instead of needing a separate world-frame projection step:
  - `uncertainty_to_radius_px(uncertainty, min_px, max_px, scale)` — clamped mapping, same
    clamp pattern `kb_teleop_vl.py` already uses for obstacles
    (`OVERLAY_MIN_PIXEL_RADIUS`/`OVERLAY_MAX_PIXEL_RADIUS`).
  - `project_body_point_to_pixel(forward, left, hfov_deg, h, w)` — body-frame point → (u, v) or
    `None` if behind/out of frame.
  - `draw_ghost_mask(rgb, u, v, radius_px, color=GREEN, alpha=...) -> np.ndarray` — translucent
    circular blend, non-destructive (returns a copy), modeled on `overlay_obstacles()`.
- Unit tests for the two pure-math pieces above (radius mapping, projection) — no sim/GPU
  needed, straightforward pytest, e.g. under a new `sam_vla/core/tests/` or alongside
  `vl_direction/tests/`'s style.

### Phase 1 — interactive verification in `kb_teleop_vl.py`

- Generalize the existing synthetic `_update_uncertainty_covariance` growth in
  [kb_teleop_vl.py:404-418](kb_teleop_vl.py:404) to call the Phase 0 growth logic (so the same
  formula is exercised here and in Phase 2, not two divergent implementations).
- After `annotated_rgb = overlay_obstacles(rgb, circles)` in `render()`
  ([kb_teleop_vl.py:507](kb_teleop_vl.py:507)), call `draw_ghost_mask(...)` using the current
  synthetic uncertainty value and a projected target point (reuse the nearest-obstacle point as
  a stand-in "goal" for this demo only — no new goal-tracking subsystem needed here, since this
  script has no real goal object). Feed the result into both the VLM dispatch
  ([kb_teleop_vl.py:519](kb_teleop_vl.py:519)) and the on-screen image
  ([kb_teleop_vl.py:526](kb_teleop_vl.py:526)) — same "one overlay, seen by both" pattern the
  file already uses for obstacles.
- No change to `_vl_worker`/`_apply_vl_line` — the direction is already advisory-only (printed +
  shown in the status line), matching the confirmed decision.
- Manual verification: run the script, confirm the green circle grows while the stand-in target
  is unseen/far and shrinks/resets when it's grounded again, stays translucent (background still
  visible through it), and the printed exploration-mode line keeps updating on the existing
  cadence.

### Phase 2 — wire into the real rollout loop (`sam_vla/run_navdp_rollout.py`)

- Pass `odom_noise_growth_rate` through as a new CLI flag alongside the existing
  `--belief-odom-noise` (wherever that's currently threaded to
  [run_navdp_rollout.py:457-465](sam_vla/run_navdp_rollout.py:457)).
- At [run_navdp_rollout.py:728-730](sam_vla/run_navdp_rollout.py:728), where `vis_rgb` is already
  built purely for visualization/logging, additionally draw the ghost mask when: single-goal mode
  (`belief_tracker is not None` — already `None` in `multi_goal`/`base_station` modes, so this
  naturally excludes the base-station DWELL/RETURN phases, which skip the policy/CBF call
  entirely at [:621-658](sam_vla/run_navdp_rollout.py:621) anyway), not currently `blocked`
  (reuse the existing `blocked` bool from [:679-685](sam_vla/run_navdp_rollout.py:679) — this is
  the "cbf vs exploration" gate), and the goal is currently unseen (`not goal_visible`).
- Dispatch `vl_direction.directive_engine.query("exploration", [vis_rgb_with_ghost],
  ExplorationContext(task_str=...), episode_id)` at a throttled cadence (mirror
  `kb_teleop_vl.py`'s every-N-frames-or-M-seconds pattern; a full VLM call every rollout step
  would be far too slow). Log the returned `Direction` into `vla_result`/`logger.log_step` next
  to the existing belief/CBF diagnostics — advisory-only, matching Phase 1.
- No change to `action`, `policy.act_verbose(...)`, or `avoidance.apply(...)` call sites.

## Open questions to settle during implementation (not blocking this plan)

- Exact `uncertainty` growth formula (linear vs. compounding via `time_since_seen`) — pick
  empirically once Phase 0's unit tests make it cheap to compare shapes; not worth debating in
  the abstract.
- Ghost-mask pixel-radius clamp bounds for Phase 2 (reuse `kb_teleop_vl.py`'s
  `OVERLAY_MIN_PIXEL_RADIUS=3` / `OVERLAY_MAX_PIXEL_RADIUS=260` as the starting point, or derive
  from `image_size` if rollouts use a different resolution).
- Whether a later phase graduates from advisory-only to actually influencing `Action`
  (proportional nudge à la `lost_goal_heading_assist`, vs. a harder override à la
  `qwen_discrete_direction_policy.direction_to_action`) — intentionally deferred; revisit only
  after Phase 1/2 visualization is validated as correct and useful on its own.

## Verification plan

- **Phase 0**: pytest on `uncertainty_to_radius_px` and `project_body_point_to_pixel` — pure
  functions, no sim dependency.
- **Phase 1**: manual visual check in the live Tkinter window (see Phase 1 bullet above).
- **Phase 2**: spot-check a handful of saved `vis_rgb` frames/video from a real rollout before
  trusting it at scale — same discipline CLAUDE.md already prescribes for segmentation-sweep
  overlays ("Sanity-check any new sweep by overlaying masks on rgb for a few frames").
