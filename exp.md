# `nav/gui.py` — the rover control console

## What it is

`NavGuiApp` is a single-process Tkinter/`customtkinter` control panel for `nav.rover_controller.RoverController`. It's the in-house equivalent of `Nav_new/MARS/launch_mars.sh`'s DINO+NavDP GUI, but drives the *published* NavDP model rather than this repo's custom S2DiT+NavDP model. It runs at a fixed ~15 Hz repaint (`REFRESH_MS = 66`), independent of the controller's own tick rate, and every frame it pulls a `d = controller.snapshot()` dataclass and re-renders the whole window from it — the GUI holds almost no state of its own beyond a handful of UI-only flags (pending click, panel visibility, manual-drive keys held, a one-line "ack" echo).

The entire layout is a **deliberate re-implementation of the REACH Handheld-Device HRI principles** (Ma et al., DLR/NASA — "Human-Robot Interaction through REACH"). Every section of code carries a `# REACH Pn:` comment naming which of ten principles it serves (P1 contextual menus, P2 three-stage command feedback, P3 chunked taskwork, P4 map/situational awareness, P5 glove-safe spacing, P6 minimal clutter, P7 point-and-select, P8 always-available emergency stop, P9 continuous feedback, P10 autonomy/intent communication). The file's own docstring is the canonical cross-reference to `reach_gui/REACH_UI_justification.md`.

## Overall window layout

The root window is a 2-column, 2-row grid, starting maximized (explicit `geometry()` fallback plus best-effort `state("zoomed")`, since not every window manager honors zoom):

- **Row 0** — the working area, split into:
  - **Column 0 (weight 3)** — the camera hero panel (live sim view).
  - **Column 1 (weight 1, min 360px)** — a scrollable sidebar containing every status panel, control, and contextual menu.
- **Row 1 (weight 0)** — a pinned emergency bar spanning both columns, *outside* the scrollable sidebar, always visible regardless of scroll position (REACH P8).

### Camera hero (left)

A `tk.Canvas` that the live RGB frame (`d.vis_rgb`) is letterboxed into every refresh, tracking whatever size the window currently is. Clicking anywhere on the rendered frame is the primary way to designate a goal — REACH P7 "point and select an object in the environment." A caption below the canvas reminds the operator of this. If a click is pending confirmation, a cyan crosshair+circle is drawn onto the frame at the clicked point (distinct from the gold marker the controller itself burns into `vis_rgb` once a goal is actually confirmed).

### Sidebar (right, scrollable, top-anchored)

A vertical stack of labeled sections, each in its own bordered/unbordered `CTkFrame`, in this order:

1. **ROVER STATUS** — always visible. Contains:
   - Three **stage pills** ("1 · RECEIVED", "2 · EXECUTING", "3 · COMPLETE") that light up in sequence (grey→cyan→green) as a command progresses — the file's implementation of REACH's three-stage command-feedback bar (P2). Stage is derived each frame from `d.goal_reached` / `d.mode` / whether the rover is currently moving, with no dedicated controller field needed.
   - A large **mode label** (color-coded per mode: idle/manual/point/resolve/review-segmentation/turn) showing the current high-level state, appending "✓ GOAL REACHED" when applicable.
   - A **detail label** that prefers the GUI's own just-pressed acknowledgment text over the controller's status text, so a button press feels instantly registered (P9) even before the controller thread's own state catches up next tick.
   - A small, grey, monospace **telemetry line** (pose, step count, velocity/yaw-rate, CBF blocked/orbit/hard-gate flags, uncertainty value vs threshold) — deliberately de-emphasized per REACH P6 (essential info only; raw numbers are secondary).
   - A hidden **"controller thread died"** warning label that only grids in if `controller.is_alive()` goes false.

2. **MAP — rover · goal · path · hazards** — a square `tk.Canvas` (`self.plot`, resizes between 200–480px with the sidebar width) drawing a body-frame top-down view centered on the rover:
   - A horizontal axis line and a small oval marking the rover itself, always at the same fixed screen position (rover-centric frame).
   - A red-outlined circle for the nearest obstacle point, if any (`d.obstacle_point`).
   - A red polyline for the recent trajectory (`d.trajectory`).
   - A gold asterisk marking the current belief-tracked goal position (`d.belief_g`).
   This is the REACH P4 "persistent situational-awareness map" — rover, goal, path, and hazards in one glance, always present regardless of mode.

3. **DESIGNATE TARGET** — the *selection* half of goal-setting (REACH P1: contextual menu shows only what's relevant right now). Four buttons plus one text entry:
   - **Segment Scene** — runs the SAM2(+LoRA)/Qwen-salience goal resolver.
   - **Ground Target** — open-vocabulary grounding via GroundingDINO, driven by the adjacent free-text entry (placeholder `"flag" / "blue cuboid"`, Enter-to-submit bound).
   - **Random Goal** — lets the rover pick a plausible exploration goal itself (P10 autonomy).
   - **Go Home** — autonomous return-to-start, same verb REACH's own rover menu uses.
   None of these buttons show the *outcome* actions (confirm/rerun/pick-manually) — those only appear in **Contextual Panel A** once a resolution actually exists, which is the point of the REACH-style adaptive menu.

4. **TASKWORK** — a free-text command box (placeholder `"go to a flag then return to home"`) plus a **Send** button. Text is handed to `RoverController.submit_nav_command`, which splits it via a Qwen VLM into an ordered list of sub-goals/directions and drives through them one at a time; a `mission_status_label` below shows N/total progress — REACH P3's "chunked, linear taskwork" so the operator can self-check without step-by-step direction.

5. **MANUAL DRIVE** — a 3×3 D-pad of large (64×52px), widely-spaced buttons (↑←→↓ only; REACH P5 glove-safe sizing/spacing) bound to press/release for continuous manual velocity while held, mirrored by the physical arrow keys (bound globally on `root`, guarded so typing in a text entry doesn't also drive the rover). Also binds numpad digit keys 1/2/3/4/6/7/8/9 to uncertainty-heading submission and `r`/`R` to retry — these are no-ops unless the uncertainty panel (below) is actually active.

6. **Contextual Panel A — REVIEW RESOLVED GOAL** *(amber border, only gridded in `MODE_REVIEW_SEGMENTATION`)* — shows the resolver's description text and three actions: **Confirm** (green, accept the goal), **Rerun** (amber, re-resolve), **Pick Manually** (fall back to point-and-select). Sidebar auto-scrolls to reveal this panel the instant it appears.

7. **Contextual Panel B — CONFIRM POINT GOAL** *(cyan border, only gridded while a camera click is pending)* — "Set the goal at the point you clicked?" with **Confirm Point Goal** (green) / **Cancel** (grey, also bound to Esc).

8. **Contextual Panel C — UNCERTAINTY HALT** *(blue border, only gridded while `d.uncertainty_halted`)* — fires when the real `BeliefGoalTracker`'s uncertainty value crosses threshold while driving toward a resolved goal. A numpad-shaped 3×3 grid of heading buttons (8 directions at 45° increments, front-relative, same convention/layout as `kb_teleop_vl.py`'s panel) lets the operator redirect the rover, plus a **Retry** button to re-request a VLM heading sweep instead of committing. Buttons disable and the title/border turn amber while a VLM request is in flight. Unlike `kb_teleop_vl.py`'s original (log-only), this panel's headings actually redirect driving.

9. **Footer** — a small grey status line (`d.click_status`) showing the outcome of the last point-goal click (e.g. "click ignored: no valid depth"); this row absorbs leftover vertical space so the whole stack stays top-anchored on tall screens instead of floating mid-window.

### Pinned emergency bar (bottom, spans both columns)

Outside the scrollable sidebar entirely, so it's reachable no matter how far the sidebar is scrolled or what mode is active (REACH P8):
- **■ EMERGENCY STOP** — large (56px), bold, red — the single most safety-critical control, given the most real estate.
- **Reset Rover** — smaller, grey, visually subordinate since it's destructive-adjacent but not an emergency action.

## Why it's built this way

The whole file is structured around one idea: **the rover gives no feedback on its own**, so the console has to manufacture every signal REACH identified operators need — command lifecycle (pills), spatial awareness (the map), safety (always-reachable stop), and low-friction acknowledgment (the `_ack_text` echo, which updates the header the instant a button is pressed, before the controller's background thread has caught up). Panels that don't apply to the current state simply aren't drawn (P1/P6), rather than being greyed out — this is what keeps a fairly feature-dense console (goal resolution, open-vocab grounding, mission taskwork, manual drive, uncertainty handling, emergency stop) from overwhelming the operator at any single moment.

Color is used consistently as the one cross-panel cue for "what is the rover doing" (`MODE_COLORS`), separate from the per-panel border colors that mean "this needs your attention now" (amber = review, cyan = confirm, blue = uncertainty halt, red = danger/stop).
