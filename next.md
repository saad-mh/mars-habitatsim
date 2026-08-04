# Plan: Base-Station Out-and-Back Rollout (hardcoded two-goal navigation)

## 0. What this is

An extension of the existing single-goal NavDP rollout (`sam_vla/run_navdp_rollout.py`) so a run does,
in order:

1. Resolve a goal from the first frame exactly as today (segmentation → one goal + the rest as
   obstacles).
2. Drive to it under NavDP + CBF, exactly as today.
3. **New:** once reached, hold position for a configurable `dwell_seconds`.
4. **New:** drive back to the rover's own spawn pose ("base station") under the same NavDP + CBF
   loop, treating the first goal as a now-passed obstacle on the way back.
5. Stop once back within success radius of the base station.

Two goals, fixed and known ahead of time (no open-vocabulary discovery involved) — `goal_1` = the
detected object, `goal_2` = the rover's own `(start_x, start_z, start_yaw_deg)`.

## 1. Is this already in the system? (short answer: half of it)

`sam_vla/run_navdp_rollout.py --multi-goal` looks like it should be the answer, but it solves a
different problem: **dynamic open-vocabulary goal discovery**. It periodically re-runs SAM3 + CLIP
(`_multi_goal_resegment`) to *mint new goal_ids at runtime* as it recognizes objects matching a
vocabulary, and only afterward hands them to `navdp.extensions.RouteManager` /
`SubgoalBeliefBank`. None of that discovery machinery is needed here — both goals are already known
before the episode starts. Bolting onto `--multi-goal` as-is would mean either faking a SAM3/CLIP
detection cycle for a goal that's actually just "the spawn point," or leaving the base station
undiscoverable by CLIP vocabulary, which is backwards.

What **is** directly reusable, because it's actually goal-id-agnostic and has no SAM3/CLIP coupling:

- **`navdp.extensions.RouteManager`** (`navdp/navdp/extensions/route_manager.py`) — an ordered,
  index-based pointer over a list of goal_ids, explicitly designed to support repeats
  (`["A", "B", "A"]` stays well-defined per its own docstring). `route=["goal_1", "base_station"]`
  is exactly what we want, with zero changes needed.
- **`navdp.extensions.SubgoalBeliefBank`** (`navdp/navdp/extensions/belief_bank.py`) — a per-goal-id
  Gaussian belief (`mu`, `Sigma`, `visible`, `confidence`) updated from a plain
  `{visible, position, confidence}` dict each step, decoupled from how that observation was produced.
  Reuse directly with `goal_ids=["goal_1", "base_station"]`; feed it the same
  `belief_tracking.mask_to_body`-derived observation already used for the single-goal path today,
  computed once per goal instead of routing through `SAMDepthTargetExtractor`.

What's genuinely missing and needs building:

- Any notion of "base station" as a goal (it's not detected, it's just the spawn pose).
- The dwell-then-return state transition — nothing in the codebase holds position for a duration and
  then re-targets. `RouteManager.update()` only advances a pointer; it doesn't pause.
- A way to keep exactly one goal painted as `MESH_GOAL_ID` at a time. `NavdpPolicy.act_verbose`
  (`sam_vla/policy/navdp_policy.py:156`) reads a **single** goal-mask channel from
  `MarsHabitatEnv.get_semantic_frame()` — it has no concept of "which of N goals." The existing
  `--multi-goal` loop dodges this by never registering scene meshes for goals at all — it manually
  paints only the active goal's last-known SAM3 mask into the semantic frame each step
  (`run_navdp_rollout.py:409-414`). We have a different, simpler tool available: our goals are static
  scene meshes (`register_object_mask`), and the returned `ManagedRigidObject` handle's
  `.semantic_id` is a plain mutable int — re-tagging it live is enough (see §3).

## 2. Base-station world position

No detection needed — it's just the spawn pose already passed to `MarsHabitatEnv`:
`(start_x, terrain_height(start_x, start_z), start_z)`. Sample the height the same way
`register_object_mask` already does internally (`terrain_patch_mesh` resamples from `self._terrain`,
i.e. `sam_vla/env/terrain.py:HeightmapGrid`) — don't hand-roll a second height lookup, since the
"Known issues" note in `CLAUDE.md` about height-normalization mismatches applies here too.

## 3. Goal-mesh bookkeeping (the part that makes single-goal-channel `NavdpPolicy` work for two goals)

Register **both** meshes once, up front, via `register_object_mask` (`sam_vla/env/habitat_env.py:267`),
keep their returned handles, and drive visibility purely through `.semantic_id` re-tagging — no
re-registration, no removal:

| mesh | at episode start | once `goal_1` is reached (→ DWELL) | once base station is reached |
|---|---|---|---|
| `goal_1` marker | `MESH_GOAL_ID` | → `MESH_OBST_ID` (still physically there, now something to steer around on the way back) | unchanged |
| `base_station` marker | `0` (neutral — invisible to both goal and obstacle channels, so it doesn't distort the outbound leg's CBF or bias the video overlay) | unchanged | → `MESH_GOAL_ID` at the *start* of RETURN, i.e. before that step's `env.get_semantic_frame()` call, so no single frame ever has two goal-tagged regions at once |

This re-tagging must happen **before** the semantic frame is fetched for the transition step, not
after — same ordering bug class as any off-by-one in a state machine driving a render.

## 4. State machine

Four phases, replacing the flat per-step loop body in `run()` for this mode:

```
OUTBOUND  -- drive toward goal_1 (identical to today's single-goal loop)
   |  route.update() reports advanced (goal_1 -> base_station)
   v
DWELL     -- hold Action(0, 0, 0) for round(dwell_seconds / dt) steps;
             re-tag goal_1 -> MESH_OBST_ID and base_station -> MESH_GOAL_ID on entry
   |  dwell counter expires
   v
RETURN    -- drive toward base_station (same NavDP + CBF loop, goal_1 now an obstacle)
   |  route.update() reports finished
   v
DONE      -- stop (mirrors today's --stop-on-route-finished)
```

`RouteManager.update(robot_position=[0,0], belief_bank=bank)` drives the OUTBOUND→DWELL and
RETURN→DONE transitions exactly as it already does in the `--multi-goal` loop
(`run_navdp_rollout.py:395`) — reused unchanged. `bank.update(...)` needs an `observations` dict each
step; for the *active* goal only, compute it from that goal's current `MESH_GOAL_ID` mask +
`belief_tracking.mask_to_body` (reuse, don't reimplement); the inactive goal gets
`{"visible": False}}` and the bank just carries its uninitialized/decayed state, which is fine since
nothing reads it until it becomes active.

`CbfObstacleAvoidance` and `safety_filter_fn` run unchanged across all four phases — the obstacle set
naturally grows to include `goal_1` once it's re-tagged, no separate wiring needed since
`obstacle_mask = semantic_render == MESH_OBST_ID` already picks up whatever currently carries that id.

During DWELL, skip the policy/CBF call entirely (hold `Action(0,0,0)`) but keep calling
`env.step(new_pose)` with the unchanged pose and keep logging, so the saved video/frames show a clean
stationary hold rather than a gap.

## 5. New config surface

CLI flags on `sam_vla/run_navdp_rollout.py` (or a thin new entry point if the flag surface gets too
tangled with `--multi-goal` — decide once §4 is actually wired up, don't pre-decide):

- `--base-station` — enables this mode. Mutually exclusive with `--multi-goal` (different problems,
  see §1 — don't try to make one subsume the other).
- `--dwell-seconds` (default e.g. `5.0`) — hold duration at goal_1.
- `--goal-success-radius` (default e.g. `1.0`) — fed to `RouteManager(success_radius=...)`. One value
  for both legs for v1; `RouteManager` only takes a single radius for the whole route today. If a
  per-goal radius turns out to matter empirically, extend it then — not preemptively.
- `--base-marker-radius` (default: reuse `--obj-mask-radius`) — size of the synthetic base-station
  disc.
- Reuse every existing `--cbf*`, `--lost-goal-*` flag unchanged.

## 6. Known risks / open questions

- **No visual target on the way back initially.** The base-station marker is a small flat disc with
  no elevation — it likely won't be visible in-frame until the rover is already close. Driving back
  relies on `SubgoalBeliefBank`'s dead-reckoned `mu` (fed by odometry between sightings, same math as
  `BeliefGoalTracker.propagate`) plus `lost_goal_heading_assist` biasing yaw/forward toward that
  belief bearing. **Recommend defaulting `--lost-goal-ghost` on for this mode** — without it, the
  return leg has no steering signal at all until the marker happens to enter frame.
- **Mesh re-tag ordering.** Get this wrong (re-tag after the frame is grabbed instead of before) and
  one step's `goal_mask` centroids over two disjoint regions, corrupting that step's belief seed. Worth
  an explicit assertion or test (see §7) rather than trusting review alone.
- **Single shared success radius** for both legs (see §5) — flagged, not solved, until there's a
  reason to solve it.
- **`goal_1`-as-obstacle proxy is a flat terrain-following disc**, same as every other obstacle bbox
  in this pipeline — consistent with existing behavior, not a new risk, just noting CBF's avoidance of
  it will be no more or less accurate than any other registered obstacle.

## 7. Validation plan before trusting output

1. Short run (`--max-steps` small, `--dwell-seconds 3`) — confirm the log shows `route_index` go
   0 → 1, a `phase` field transition OUTBOUND → DWELL → RETURN → DONE, and the dwell duration matches
   `round(3 / dt)` steps of zero-velocity actions.
2. Confirm the logged base-station world position matches `(start_x, start_z)` from the CLI args.
3. Inspect the saved video/frame overlays: `goal_1`'s marker should render obstacle-colored (not
   goal-colored) for every RETURN-phase frame, and the base marker should render goal-colored only
   from DWELL onward, never during OUTBOUND.
4. Confirm final logged `distance_to_goal(final_pose, base_position)` is within
   `--goal-success-radius`.
5. Re-run the existing single-goal path (`--base-station` unset) afterward and confirm its output is
   byte-for-byte unchanged in behavior — this must be strictly additive, same guarantee the
   `--multi-goal` flag already gives (`run_navdp_rollout.py:232-234`'s comment: unset, behavior is
   unchanged).
