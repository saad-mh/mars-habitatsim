#!/usr/bin/env python3
"""Interactive control panel for nav.rover_controller.RoverController --
the in-house, single-process equivalent of Nav_new/MARS/launch_mars.sh's
DINO+NavDP GUI, driving the actual published NavDP model instead of this
repo's own custom S2DiT+NavDP model (see rover_controller.py's docstring).

Run via nav/launch_nav.sh, or directly:
    conda activate habitat
    cd mars-habitatsim
    python -m nav.gui [--scene-path ...] [--heightmap-path ...] [--cbf/--no-cbf] ...

Open-vocabulary target grounding ("Ground Target" / a Command-panel GO_TO/FIND
step) uses GroundingDINO (navdp.extensions.GroundingDINODetector, see
sam_vla.goal_resolution.dino_grounding_resolver) instead of Qwen VLM
point-grounding -- ported from scripts/vlm_nav_tests/qwen_search_dino.py's
own DINO usage, --dino-* flags below configure it. The SAM2+Qwen-salience
"Segment" path and the Qwen-based free-text Command-splitting/uncertainty-halt
prompts are unrelated and still use Qwen.

========================================================================
REACH UI/UX PRINCIPLES  (Ma et al., "Human-Robot Interaction through
REACH", DLR/NASA) -- this layout is a deliberate re-implementation of the
REACH Handheld-Device interface principles for a desktop sim console.
Every control below carries a `# REACH Pn:` tag naming the principle it
serves; the same numbering is used in reach_gui/REACH_UI_justification.md.

  P1  CONTEXTUAL / ADAPTIVE MENUS. Actions are shown *because of* the
      current selection or rover state, not all at once. (REACH: "a
      contextual menu that adapts based on user selection" -- object vs
      module vs rover give different actions.)
  P2  THREE-STAGE COMMAND FEEDBACK. Every command surfaces an explicit
      lifecycle: RECEIVED -> EXECUTING -> COMPLETE. (REACH rover-status
      bar: "1) acknowledgment of command receipt, 2) task execution
      status, 3) task completion confirmation" -- the rover gives no
      other signal.)
  P3  CHUNKED, LINEAR TASKWORK. Multi-step work is shown as an ordered
      list with N/total progress so the operator can self-check without
      line-by-line direction. (REACH taskwork pages, "feasible chunks".)
  P4  MAP / SITUATIONAL AWARENESS. A persistent frame of reference:
      rover pose, goal, planned path and hazards, available at any
      moment. (REACH map: rover + user position, terrain/hazards,
      suggested path.)
  P5  SIMPLIFIED, GLOVE-SAFE CONTROLS. Few, large, well-separated
      buttons to prevent accidental activation. (REACH: buttons "spaced
      ... further apart to prevent accidental activation, when wearing
      thick gloves"; "controls were simplified".)
  P6  ONLY ESSENTIAL INFORMATION. Minimize clutter / cognitive load;
      advanced detail is de-emphasized or revealed on demand. (REACH:
      "only essential information is exchanged"; reduced cognitive load.)
  P7  INTUITIVE POINT-AND-SELECT. Direct the rover by pointing at a
      target in the live view. (REACH: "pointing and selecting ... an
      object in environment".)
  P8  ALWAYS-AVAILABLE SAFETY / EMERGENCY TAKEOVER. Stop is reachable at
      all times regardless of scroll or mode. (REACH Rover-Driver
      Console: "emergency takeover"; proximity safety.)
  P9  CONTINUOUS FEEDBACK, LOW FRUSTRATION. Immediate acknowledgment of
      every action to "curb frustration/confusion". (REACH results:
      -40% frustration driven by continuous feedback.)
  P10 AUTONOMY & INTENT COMMUNICATION. Show what the rover is doing and
      why (mode, goal, progress), supporting perceived autonomy. (REACH:
      "communicate intents and progress for better HRC".)
========================================================================
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk

from sam_vla.goal_resolution import dino_grounding_resolver
from vl_direction import config as vl_dir_config

from nav.rover_controller import MODE_REVIEW_SEGMENTATION, MODE_TURN, RoverController

REPO_ROOT = Path(__file__).resolve().parent.parent
REFRESH_MS = 66  # ~15 Hz GUI repaint, independent of the controller's own hz

PLOT_BG = "#242424"
PLOT_AXIS = "#555555"
PLOT_ROVER = "#e5e7eb"
PLOT_OBSTACLE = "#f87171"
PLOT_TRAJECTORY = "#f97373"
PLOT_GOAL = "#fbbf24"

# REACH P5: glove-safe control sizing. One place to tune the whole
# console's button footprint so every primary action stays large and
# every gap stays wide enough to miss with a thick glove -- the paper's
# core hardware fix ("spaced the buttons further apart").
BTN_H = 46  # primary action button height (px)
BTN_GAP = 8  # inter-button spacing (px)
SECTION_GAP = 12  # spacing between labeled sections (px)

# REACH P2: the three-stage command lifecycle, in order. The status
# header renders exactly these, lighting the active one -- this is the
# whole "rover status update bar" of the paper, made unmissable.
STAGE_ORDER = ("received", "executing", "complete")
STAGE_LABELS = {
    "received": "1 · RECEIVED",
    "executing": "2 · EXECUTING",
    "complete": "3 · COMPLETE",
}
STAGE_ACTIVE_COLOR = "#22d3ee"
STAGE_DONE_COLOR = "#4ade80"
STAGE_IDLE_COLOR = "#3f3f46"

# Mode -> accent color, shared by the big mode label and (where relevant)
# the panel that goes with that mode -- REACH P10: keeps color the one
# consistent cue for "what is the rover doing right now" across the
# whole window.
MODE_COLORS = {
    "idle": "#9ca3af",
    "manual": "#60a5fa",
    "point": "#4ade80",
    "resolve": "#c084fc",
    MODE_REVIEW_SEGMENTATION: "#f59e0b",
    MODE_TURN: "#fb923c",
}

# Uncertainty-halt numpad panel: rover-front-relative headings (degrees,
# same convention as nav/goal_math.heading_ahead_point -- 0=front,
# positive=clockwise) bound to the same physical key layout
# scripts/habitat_tests/kb_teleop_vl.py's uncertainty panel uses, redrawn
# independently here rather than imported (nav/ stays decoupled from that
# script). Grid position drives the heads-up panel's clickable button
# layout, keys drive the numpad keyboard shortcuts.
UNCERTAINTY_HEADING_KEYS = {
    "8": 0.0,
    "9": 45.0,
    "6": 90.0,
    "3": 135.0,
    "2": 180.0,
    "1": -135.0,
    "4": -90.0,
    "7": -45.0,
}
UNCERTAINTY_HEADING_GRID_POS = {
    "7": (0, 0),
    "8": (0, 1),
    "9": (0, 2),
    "4": (1, 0),
    "6": (1, 2),
    "1": (2, 0),
    "2": (2, 1),
    "3": (2, 2),
}


def _default_navdp_upstream_ckpt() -> Optional[str]:
    """Same repo-relative fallback sam_vla.run_navdp_rollout uses -- a plain
    invocation just works on a checkout that already has Phase 0's
    (next.md's Integration project) gitignored checkpoint in place."""
    candidate = REPO_ROOT / "navdp" / "navdp-cross-modal.ckpt"
    return str(candidate) if candidate.exists() else None


class NavGuiApp:
    SIDEBAR_MIN_W = 360
    PLOT_MIN = 200
    PLOT_MAX = 480
    PLOT_RANGE = 6.0  # meters shown top-to-bottom of the body-frame plot

    def __init__(
        self,
        root: ctk.CTk,
        controller: RoverController,
        max_linear: float,
        max_angular: float,
    ):
        self.root = root
        self.controller = controller
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.closed = False
        self._manual_held: set[str] = set()
        self._pending_click_norm: Optional[tuple[float, float]] = None
        self._seg_panel_visible = False
        self._click_panel_visible = False
        self._cam_img_box: Optional[tuple[int, int, int, int]] = None
        self._cam_photo = None
        self._plot_size = self.PLOT_MIN
        # REACH P9: last action acknowledged, echoed in the status header
        # the instant a button is pressed (before the controller's own
        # state catches up on the next tick) so no press ever feels lost.
        self._ack_text = ""

        root.title("mars-habitatsim/nav · REACH console")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.minsize(900, 640)

        # Start maximized so the layout below (sized entirely off grid
        # weights, not fixed pixel panels) actually gets the whole screen to
        # lay out into. Always set an explicit near-full-screen geometry
        # first -- window managers that ignore "zoomed"/"-zoomed" (or a
        # WM-less display) would otherwise silently leave the window at its
        # small natural size -- then best-effort request a true maximize on
        # top of that for the window managers that do support it.
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{int(sw * 0.95)}x{int(sh * 0.9)}")
        try:
            root.state("zoomed")
        except tk.TclError:
            try:
                root.attributes("-zoomed", True)
            except tk.TclError:
                pass

        # Root grid: row 0 is the working area (camera hero + sidebar),
        # row 1 is the pinned emergency bar. REACH P8: the STOP control
        # lives OUTSIDE the scrollable sidebar, on its own always-visible
        # root row, so an emergency takeover is one click away no matter
        # how far the sidebar is scrolled or what mode is active.
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=0)
        root.grid_columnconfigure(0, weight=3)
        root.grid_columnconfigure(1, weight=1, minsize=self.SIDEBAR_MIN_W)

        mono_font = ctk.CTkFont(family="Consolas", size=12)
        mode_font = ctk.CTkFont(size=17, weight="bold")

        # ---------------- camera hero (REACH P7: point-and-select) ------- #
        # Fills all left-column space; the frame is letterboxed into it on
        # every refresh so it tracks window size. Clicking the live view is
        # the primary, most intuitive way to designate a goal -- the direct
        # analog of REACH "point and select an object in the environment".
        cam_frame = ctk.CTkFrame(root)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        cam_frame.grid_rowconfigure(0, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        self.cam_canvas = tk.Canvas(cam_frame, bg="#111111", highlightthickness=0)
        self.cam_canvas.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 2))
        self.cam_canvas.bind("<Button-1>", self.on_cam_click)
        ctk.CTkLabel(
            cam_frame,
            text="POINT & SELECT — click any point in the live view to send the rover there",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#22d3ee",
        ).grid(row=1, column=0, pady=(0, 6))

        # ---------------- sidebar ---------------------------------------- #
        # REACH P6 (only essential info) + P5 (spacing): one vertical
        # column of clearly-labeled sections, top-anchored, generous gaps.
        # Scrollable so no control is ever clipped off a short monitor
        # (contextual review panels can push the stack past screen height).
        sidebar = ctk.CTkScrollableFrame(root, fg_color="transparent")
        sidebar.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=(10, 4))
        sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar = sidebar

        row = 0

        # ===== SECTION 1 — ROVER STATUS (REACH P2 + P9 + P10) =========== #
        # The single most important panel in the paper: continuous,
        # three-stage feedback the rover cannot otherwise provide. Placed
        # first and always visible.
        self.status_panel = ctk.CTkFrame(
            sidebar, border_width=2, border_color="#3f3f46"
        )
        self.status_panel.grid(row=row, column=0, sticky="ew")
        row += 1
        self.status_panel.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self.status_panel,
            text="ROVER STATUS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#9ca3af",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))

        # REACH P2: the three lifecycle pills, lit in order.
        stage_row = ctk.CTkFrame(self.status_panel, fg_color="transparent")
        stage_row.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        for i in range(3):
            stage_row.grid_columnconfigure(i, weight=1)
        self.stage_pills: dict[str, ctk.CTkLabel] = {}
        for i, stage in enumerate(STAGE_ORDER):
            pill = ctk.CTkLabel(
                stage_row,
                text=STAGE_LABELS[stage],
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color=STAGE_IDLE_COLOR,
                corner_radius=6,
                text_color="#e5e7eb",
            )
            pill.grid(row=0, column=i, sticky="ew", padx=2, ipady=5)
            self.stage_pills[stage] = pill

        self.mode_label = ctk.CTkLabel(
            self.status_panel, text="", anchor="w", justify="left", font=mode_font
        )
        self.mode_label.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 0))
        self.detail_label = ctk.CTkLabel(
            self.status_panel,
            text="",
            anchor="w",
            justify="left",
            wraplength=self.SIDEBAR_MIN_W - 40,
            font=ctk.CTkFont(size=12),
        )
        self.detail_label.grid(row=3, column=0, sticky="ew", padx=10, pady=(2, 2))
        self.status_panel.bind(
            "<Configure>",
            lambda e: self.detail_label.configure(wraplength=max(200, e.width - 20)),
        )
        # REACH P6: verbose numeric telemetry is deliberately de-emphasized
        # (small, grey, mono) -- present for an operator who wants it, but
        # never competing with the primary status above.
        self.telemetry_label = ctk.CTkLabel(
            self.status_panel,
            text="",
            anchor="w",
            justify="left",
            font=mono_font,
            text_color="#9ca3af",
        )
        self.telemetry_label.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 8))
        self.alive_label = ctk.CTkLabel(
            self.status_panel,
            text="controller thread died -- see console",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#f87171",
        )
        self._alive_label_visible = False

        # ===== SECTION 2 — MAP / SITUATIONAL AWARENESS (REACH P4) ======= #
        # Persistent body-frame frame-of-reference: rover, goal, planned
        # path (trajectory) and hazards (obstacles) always on screen.
        plot_frame = ctk.CTkFrame(sidebar)
        plot_frame.grid(row=row, column=0, sticky="ew", pady=(SECTION_GAP, 0))
        row += 1
        plot_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            plot_frame,
            text="MAP — rover · goal · path · hazards",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#9ca3af",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(6, 0))
        self.plot = tk.Canvas(
            plot_frame, height=self.PLOT_MIN, bg=PLOT_BG, highlightthickness=0
        )
        self.plot.grid(row=1, column=0, sticky="ew", padx=4, pady=(2, 4))
        self.plot.bind("<Configure>", self._on_plot_configure)

        # ===== SECTION 3 — DESIGNATE TARGET (REACH P1 + P7) ============= #
        # REACH P1 (contextual menu): this section is only the *selection*
        # half of the workflow -- the three ways to designate a target.
        # The *action* half (Confirm / Rerun / Pick manually, or a point
        # confirm) is not shown here; it appears contextually below only
        # once a target actually exists. That mirrors REACH's "select an
        # object, THEN the relevant actions appear" adaptive menu.
        target = ctk.CTkFrame(sidebar)
        target.grid(row=row, column=0, sticky="ew", pady=(SECTION_GAP, 0))
        row += 1
        target.grid_columnconfigure(0, weight=1)
        target.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            target, text="DESIGNATE TARGET", font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))

        # REACH P7/P1: segment the scene and let the model pick a rock goal.
        ctk.CTkButton(
            target, text="Segment Scene", height=BTN_H, command=self.resolve_goal
        ).grid(row=1, column=0, sticky="ew", padx=(10, 4), pady=BTN_GAP // 2)
        # REACH P1: open-vocabulary grounding -- name a target the segmenter
        # wasn't trained on; produces the same reviewable goal.
        ctk.CTkButton(
            target, text="Ground Target", height=BTN_H, command=self.ground_target
        ).grid(row=1, column=1, sticky="ew", padx=(4, 10), pady=BTN_GAP // 2)
        # REACH P1: free-text open-vocabulary entry feeding Ground Target.
        self.ground_entry = ctk.CTkEntry(
            target, placeholder_text='name it: "flag" / "blue cuboid"', height=38
        )
        self.ground_entry.grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, BTN_GAP)
        )
        self.ground_entry.bind("<Return>", lambda e: self.ground_target())
        # REACH P10 (autonomy): let the rover self-select a plausible
        # exploration goal when the operator has no specific target.
        ctk.CTkButton(
            target, text="Random Goal", height=BTN_H, command=self.random_goal
        ).grid(row=3, column=0, sticky="ew", padx=(10, 4), pady=(0, BTN_GAP))
        # REACH P1 (rover-as-selectable-object → "go home" action): the
        # paper's rover menu includes "go home" (autonomous return to
        # start). Same verb, same place.
        ctk.CTkButton(target, text="Go Home", height=BTN_H, command=self.go_home).grid(
            row=3, column=1, sticky="ew", padx=(4, 10), pady=(0, BTN_GAP)
        )

        # ===== SECTION 4 — TASKWORK / MISSION (REACH P3 + P10) ========== #
        # Free-text command -> ordered, chunked mission the operator can
        # self-check step by step (progress synced from d.mission_status),
        # exactly REACH's linear taskwork pages that "reduce the need for
        # constant, line-by-line directions from ground control".
        command_panel = ctk.CTkFrame(sidebar)
        command_panel.grid(row=row, column=0, sticky="ew", pady=(SECTION_GAP, 0))
        row += 1
        command_panel.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            command_panel, text="TASKWORK", font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))
        self.command_entry = ctk.CTkEntry(
            command_panel,
            placeholder_text='"go to a flag then return to home"',
            height=38,
        )
        self.command_entry.grid(row=1, column=0, sticky="ew", padx=(10, 4), pady=(0, 4))
        self.command_entry.bind("<Return>", lambda e: self.submit_command())
        # REACH P3: dispatch the ordered mission.
        ctk.CTkButton(
            command_panel, text="Send", width=72, height=38, command=self.submit_command
        ).grid(row=1, column=1, sticky="ew", padx=(4, 10), pady=(0, 4))
        self.mission_status_label = ctk.CTkLabel(
            command_panel,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color="#e5e7eb",
            wraplength=self.SIDEBAR_MIN_W - 20,
        )
        self.mission_status_label.grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, BTN_GAP)
        )

        # ===== SECTION 5 — MANUAL DRIVE (REACH P5 + P7) ================= #
        # Direct D-pad control, large glove-safe keys, always visible.
        drive = ctk.CTkFrame(sidebar)
        drive.grid(row=row, column=0, sticky="ew", pady=(SECTION_GAP, 0))
        row += 1
        drive.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            drive, text="MANUAL DRIVE", font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 4))
        dpad = ctk.CTkFrame(drive, fg_color="transparent")
        dpad.grid(row=1, column=0, pady=(0, 4))
        dpad_cells = {
            "fwd": (0, 1, "↑"),
            "left": (1, 0, "←"),
            "right": (1, 2, "→"),
            "back": (2, 1, "↓"),
        }
        # REACH P5: each drive key is large (64x52) and separated by wide
        # padding so a gloved press cannot spill onto the neighbour.
        for direction, (r, c, glyph) in dpad_cells.items():
            b = ctk.CTkButton(dpad, text=glyph, width=64, height=52, font=mode_font)
            b.bind("<ButtonPress-1>", lambda e, d=direction: self.manual_press(d))
            b.bind("<ButtonRelease-1>", lambda e, d=direction: self.manual_release(d))
            b.grid(row=r, column=c, padx=6, pady=6)
        ctk.CTkLabel(
            drive,
            text="hold a direction, or use the arrow keys — Esc cancels a pending click",
            font=ctk.CTkFont(size=11),
            text_color="#9ca3af",
            wraplength=self.SIDEBAR_MIN_W - 40,
            justify="left",
        ).grid(row=2, column=0, sticky="w", padx=10, pady=(0, BTN_GAP))

        # Global keyboard shortcuts below are bound on root, so they fire on
        # every keypress in the window regardless of focus -- guard each one
        # against a text entry having focus, or typing e.g. "left" or a
        # digit into it would also drive the rover / submit a heading.
        for key, direction in (
            ("Up", "fwd"),
            ("Down", "back"),
            ("Left", "left"),
            ("Right", "right"),
        ):
            root.bind(
                f"<KeyPress-{key}>",
                lambda e, d=direction: self._guarded(self.manual_press, d),
            )
            root.bind(
                f"<KeyRelease-{key}>",
                lambda e, d=direction: self._guarded(self.manual_release, d),
            )
        root.bind("<Escape>", lambda e: self.clear_all())

        # Numpad heading shortcuts for the uncertainty-halt panel below --
        # bound unconditionally (these digits aren't used elsewhere in this
        # GUI); the controller itself drops a submission that isn't
        # currently halted, so this is a no-op the rest of the time.
        for key, angle in UNCERTAINTY_HEADING_KEYS.items():
            root.bind(
                f"<KeyPress-{key}>",
                lambda e, a=angle: self._guarded(self.submit_uncertainty_heading, a),
            )
        root.bind("<KeyPress-r>", lambda e: self._guarded(self.retry_uncertainty))
        root.bind("<KeyPress-R>", lambda e: self._guarded(self.retry_uncertainty))

        # ===== CONTEXTUAL PANEL A — REVIEW RESOLVED GOAL (REACH P1) ===== #
        # Only gridded while actually reviewing a resolved goal. This IS a
        # REACH contextual menu: the Confirm / Rerun / Pick-manually
        # actions exist only because a segmentation/grounding selection
        # exists to act on. Bordered so it reads as "action needed".
        self._seg_row = row
        row += 1
        self.seg_panel = ctk.CTkFrame(sidebar, border_width=2, border_color="#f59e0b")
        ctk.CTkLabel(
            self.seg_panel,
            text="REVIEW RESOLVED GOAL",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#f59e0b",
        ).pack(pady=(10, 2))
        self.seg_desc_label = ctk.CTkLabel(
            self.seg_panel,
            text="",
            wraplength=self.SIDEBAR_MIN_W - 40,
            justify="left",
        )
        self.seg_desc_label.pack(padx=12, pady=(0, 8), fill="x")
        self.seg_panel.bind(
            "<Configure>",
            lambda e: self.seg_desc_label.configure(wraplength=max(200, e.width - 24)),
        )
        seg_btn_col = ctk.CTkFrame(self.seg_panel, fg_color="transparent")
        seg_btn_col.pack(pady=(0, 12), fill="x", padx=12)
        seg_btn_col.grid_columnconfigure(0, weight=1)
        # REACH P1: accept this selection as the goal.
        ctk.CTkButton(
            seg_btn_col,
            text="Confirm",
            height=BTN_H,
            command=self.confirm_segmentation,
            fg_color="#15803d",
            hover_color="#166534",
        ).grid(row=0, column=0, sticky="ew", pady=3)
        # REACH P1: reject and re-resolve (same intent, new result).
        ctk.CTkButton(
            seg_btn_col,
            text="Rerun",
            height=BTN_H,
            command=self.rerun_segmentation,
            fg_color="#b45309",
            hover_color="#92400e",
        ).grid(row=1, column=0, sticky="ew", pady=3)
        # REACH P1 + P7: fall back to point-and-select if the auto result
        # is wrong.
        ctk.CTkButton(
            seg_btn_col, text="Pick Manually", height=BTN_H, command=self.pick_manually
        ).grid(row=2, column=0, sticky="ew", pady=3)

        # ===== CONTEXTUAL PANEL B — CONFIRM POINT GOAL (REACH P1 + P7) == #
        # Only gridded while a click is pending confirmation -- the
        # contextual action menu for a point-and-select target.
        self._click_row = row
        row += 1
        self.click_panel = ctk.CTkFrame(sidebar, border_width=2, border_color="#00b8d4")
        self.click_desc_label = ctk.CTkLabel(
            self.click_panel,
            text="Set the goal at the point you clicked?",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#22d3ee",
            wraplength=self.SIDEBAR_MIN_W - 40,
        )
        self.click_desc_label.pack(pady=(10, 6), padx=12)
        click_btn_col = ctk.CTkFrame(self.click_panel, fg_color="transparent")
        click_btn_col.pack(pady=(0, 10), fill="x", padx=12)
        click_btn_col.grid_columnconfigure(0, weight=1)
        # REACH P7: commit the pointed-at goal.
        ctk.CTkButton(
            click_btn_col,
            text="Confirm Point Goal",
            height=BTN_H,
            command=self.confirm_pixel_goal,
            fg_color="#15803d",
            hover_color="#166534",
        ).grid(row=0, column=0, sticky="ew", pady=3)
        # REACH P1: back out of the selection (REACH menu "back button").
        ctk.CTkButton(
            click_btn_col,
            text="Cancel",
            height=BTN_H,
            command=self.cancel_pixel_click,
            fg_color="#4b5563",
            hover_color="#374151",
        ).grid(row=1, column=0, sticky="ew", pady=3)

        # ===== CONTEXTUAL PANEL C — UNCERTAINTY HALT (REACH P1+P8+P10) == #
        # Only gridded while the controller is halted on belief
        # uncertainty. A contextual heading-request menu: the rover has
        # stopped itself (P8 safety) and asks the human for a direction
        # (P10 shared autonomy). Same bordered numpad shape as
        # scripts/habitat_tests/kb_teleop_vl.py's panel, redrawn here.
        self._uncertainty_row = row
        row += 1
        self.uncertainty_panel = ctk.CTkFrame(
            sidebar, border_width=2, border_color="#38bdf8"
        )
        self.uncertainty_title = ctk.CTkLabel(
            self.uncertainty_panel, text="", font=ctk.CTkFont(size=14, weight="bold")
        )
        self.uncertainty_title.pack(pady=(10, 2))
        self.uncertainty_desc_label = ctk.CTkLabel(
            self.uncertainty_panel,
            text="",
            wraplength=self.SIDEBAR_MIN_W - 40,
            justify="left",
        )
        self.uncertainty_desc_label.pack(padx=12, pady=(0, 8), fill="x")
        self.uncertainty_panel.bind(
            "<Configure>",
            lambda e: self.uncertainty_desc_label.configure(
                wraplength=max(200, e.width - 24)
            ),
        )
        uncertainty_grid = ctk.CTkFrame(self.uncertainty_panel, fg_color="transparent")
        uncertainty_grid.pack(pady=(0, 6))
        self.uncertainty_buttons: dict[str, ctk.CTkButton] = {}
        # REACH P1 + P5: one large heading key per compass direction, in
        # the numpad layout, spaced to prevent mis-selection.
        for key, angle in UNCERTAINTY_HEADING_KEYS.items():
            r, c = UNCERTAINTY_HEADING_GRID_POS[key]
            btn = ctk.CTkButton(
                uncertainty_grid,
                text=f"{key}\n{angle:+.0f}°",
                width=68,
                height=50,
                command=lambda a=angle: self.submit_uncertainty_heading(a),
            )
            btn.grid(row=r, column=c, padx=4, pady=4)
            self.uncertainty_buttons[key] = btn
        # REACH P1: re-request a VLM heading sweep instead of committing.
        self.uncertainty_retry_button = ctk.CTkButton(
            self.uncertainty_panel,
            text="Retry (R)",
            height=BTN_H,
            fg_color="#b45309",
            hover_color="#92400e",
            command=self.retry_uncertainty,
        )
        self.uncertainty_retry_button.pack(pady=(0, 12))
        self._uncertainty_panel_visible = False

        # ===== FOOTER — last click outcome (REACH P9) =================== #
        # Persistent one-line acknowledgment of the last point-goal click
        # ("point goal set at ..." / "click ignored: no valid depth").
        # This row absorbs leftover sidebar height so the stack stays
        # top-anchored instead of floating mid-window.
        footer_row = row
        sidebar.grid_rowconfigure(footer_row, weight=1)
        self.click_status_label = ctk.CTkLabel(
            sidebar,
            text="",
            anchor="n",
            font=ctk.CTkFont(size=11),
            text_color="#9ca3af",
            wraplength=self.SIDEBAR_MIN_W - 20,
        )
        self.click_status_label.grid(
            row=footer_row, column=0, sticky="new", padx=4, pady=(BTN_GAP, 0)
        )

        # ===== PINNED EMERGENCY BAR (REACH P8) ========================== #
        # On its own root row, spanning the full width, OUTSIDE the
        # scrollable sidebar -- always on screen. This is the paper's
        # emergency-takeover affordance: the single most safety-critical
        # control gets the largest, reddest, most reachable real estate.
        stop_bar = ctk.CTkFrame(root, fg_color="transparent")
        stop_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        stop_bar.grid_columnconfigure(0, weight=3)
        stop_bar.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            stop_bar,
            text="■  EMERGENCY STOP",
            height=56,
            font=ctk.CTkFont(size=18, weight="bold"),
            command=self.stop,
            fg_color="#b91c1c",
            hover_color="#991b1b",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        # REACH P8: reset/recover is destructive-adjacent, so it sits next
        # to STOP but is visually subordinate (grey, narrower).
        ctk.CTkButton(
            stop_bar,
            text="Reset Rover",
            height=56,
            command=self.reset_rover,
            fg_color="#4b5563",
            hover_color="#374151",
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.root.after(REFRESH_MS, self.refresh)

    # ---------------- dynamic sizing ---------------- #
    def _scroll_sidebar_to_bottom(self) -> None:
        # The contextual review/confirm panels grid in near the bottom of
        # the sidebar stack; on a short window the sidebar scrolls rather
        # than clipping, so jump the scroll down as soon as one appears
        # instead of leaving an action-required panel to be found by
        # accident (REACH P9: never make the operator hunt for the next
        # step). Deferred one tick since the panel's own .grid() hasn't been
        # laid out (scrollregion hasn't grown to include it) yet this frame.
        self.root.after_idle(lambda: self.sidebar._parent_canvas.yview_moveto(1.0))

    def _scroll_sidebar_to_top(self) -> None:
        # Mirror of _scroll_sidebar_to_bottom for when a contextual panel
        # closes -- return to the always-on status/map sections above.
        self.root.after_idle(lambda: self.sidebar._parent_canvas.yview_moveto(0.0))

    def _on_plot_configure(self, event) -> None:
        # Keep the body-frame canvas square, tracking the sidebar's current
        # width (clamped) instead of a fixed pixel size baked in at startup.
        new_size = max(self.PLOT_MIN, min(int(event.width), self.PLOT_MAX))
        if abs(new_size - self._plot_size) > 2:
            self._plot_size = new_size
            self.plot.configure(height=new_size)

    # ---------------- commands ---------------- #
    def _guarded(self, fn, *fn_args) -> None:
        # Skip root-level keyboard shortcuts while a text entry has focus,
        # so typing there doesn't also drive the rover / trigger a heading
        # submission (see the binding loops above).
        if self.root.focus_get() in (self.command_entry, self.ground_entry):
            return
        fn(*fn_args)

    def _ack(self, text: str) -> None:
        # REACH P9: record an immediate human-readable acknowledgment of
        # the just-pressed control; refresh() echoes it in the status
        # header so no press ever feels unregistered.
        self._ack_text = text

    def submit_command(self) -> None:
        # Handed to RoverController.submit_nav_command, which splits it via
        # the Qwen VLM (qwen_client.parse_nav_command) into (directions,
        # goals) on a background thread, turns that into an ordered Mission
        # (nav.mission.parse_parts), and drives through its sub-goals one at
        # a time. Progress shows in mission_status_label.
        text = self.command_entry.get().strip()
        if not text:
            return
        print(f"[nav command] submitted: {text}")
        self._ack(f"taskwork queued: {text}")
        self.controller.submit_nav_command(text)
        self.command_entry.delete(0, "end")

    def resolve_goal(self) -> None:
        self._manual_held.clear()
        self.cancel_pixel_click()
        self._ack("segmenting scene…")
        self.controller.request_resolve()

    def ground_target(self) -> None:
        text = self.ground_entry.get().strip()
        if not text:
            return
        self._manual_held.clear()
        self.cancel_pixel_click()
        self._ack(f"grounding target: {text}")
        self.controller.request_resolve(target_text=text)

    def random_goal(self) -> None:
        self._manual_held.clear()
        self.cancel_pixel_click()
        self._ack("random goal requested")
        self.controller.random_goal()

    def go_home(self) -> None:
        self._manual_held.clear()
        self.cancel_pixel_click()
        self._ack("returning home")
        self.controller.go_home()

    def reset_rover(self) -> None:
        self._manual_held.clear()
        self.cancel_pixel_click()
        self._ack("rover reset")
        self.controller.request_reset()

    def stop(self) -> None:
        self._manual_held.clear()
        self.cancel_pixel_click()
        self._ack("STOP — driving halted")
        self.controller.stop_driving()

    def clear_all(self) -> None:
        # Escape key: a hard clear, not just stop() -- also strips resolved
        # goal/obstacle masks, the active mission, and any uncertainty halt,
        # leaving the rover exactly where it is (unlike reset_rover, which
        # also teleports it back to spawn). See RoverController.clear_all_goals.
        self._manual_held.clear()
        self.cancel_pixel_click()
        self._ack("cleared — all goals, missions, and masks removed")
        self.controller.clear_all_goals()

    # ---------------- segmentation review (Goal 1) ---------------- #
    def confirm_segmentation(self) -> None:
        self._ack("goal confirmed")
        self.controller.request_confirm_segmentation()

    def rerun_segmentation(self) -> None:
        self._ack("re-resolving goal…")
        self.controller.request_rerun_segmentation()

    def pick_manually(self) -> None:
        self._manual_held.clear()
        self._ack("pick a point in the live view")
        self.controller.request_pick_manually()

    # ---------------- uncertainty halt ---------------- #
    def submit_uncertainty_heading(self, angle_deg: float) -> None:
        self._ack(f"heading sent: {angle_deg:+.0f}°")
        self.controller.submit_uncertainty_heading(angle_deg)

    def retry_uncertainty(self) -> None:
        self._ack("requesting VLM sweep…")
        self.controller.retry_uncertainty_request()

    # ---------------- click-to-goal ---------------- #
    def on_cam_click(self, event) -> None:
        # The camera frame is letterboxed into cam_canvas (see _draw_camera),
        # so a click has to be mapped through the last-drawn image's actual
        # on-canvas box, not the canvas's own (possibly wider/taller) bounds.
        if self._cam_img_box is None:
            return
        x0, y0, x1, y1 = self._cam_img_box
        if not (x0 <= event.x <= x1 and y0 <= event.y <= y1):
            return
        nx = min(max((event.x - x0) / (x1 - x0), 0.0), 1.0)
        ny = min(max((event.y - y0) / (y1 - y0), 0.0), 1.0)
        self._pending_click_norm = (nx, ny)
        self._ack("point selected — confirm below")

    def confirm_pixel_goal(self) -> None:
        if self._pending_click_norm is None:
            return
        self._manual_held.clear()
        nx, ny = self._pending_click_norm
        self._ack("point goal committed")
        self.controller.request_pixel_goal(nx, ny)
        self._pending_click_norm = None

    def cancel_pixel_click(self) -> None:
        self._pending_click_norm = None

    # ---------------- manual drive ---------------- #
    def manual_press(self, direction: str) -> None:
        self.cancel_pixel_click()
        self._manual_held.add(direction)
        self._manual_update()

    def manual_release(self, direction: str) -> None:
        self._manual_held.discard(direction)
        self._manual_update()

    def _manual_update(self) -> None:
        if not self._manual_held:
            self.controller.stop_driving()
            return
        lin = ang = 0.0
        if "fwd" in self._manual_held:
            lin += self.max_linear
        if "back" in self._manual_held:
            lin -= 0.5 * self.max_linear
        if "left" in self._manual_held:
            ang += self.max_angular
        if "right" in self._manual_held:
            ang -= self.max_angular
        self.controller.set_manual(lin, ang)

    def on_close(self) -> None:
        self.closed = True
        self.root.destroy()

    # ---------------- refresh loop ---------------- #
    def _derive_stage(self, d) -> Optional[str]:
        # REACH P2: fold the controller's existing signals into the
        # three-stage lifecycle without needing any new controller field.
        # COMPLETE wins; otherwise an active goal that is moving is
        # EXECUTING, an active goal not yet moving is RECEIVED. No goal ->
        # no stage lit (idle / manual).
        if d.goal_reached:
            return "complete"
        active = d.mode in ("point", "resolve", MODE_TURN, MODE_REVIEW_SEGMENTATION)
        if not active:
            return None
        moving = abs(d.action.v_fwd) > 0.01 or abs(d.action.yaw_rate) > 0.01
        return "executing" if moving else "received"

    def _sync_stage_pills(self, stage: Optional[str]) -> None:
        # Light stages up to and including the active one: past stages
        # green (done), current stage cyan (active), future stages grey.
        active_idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else -1
        for i, name in enumerate(STAGE_ORDER):
            if active_idx < 0:
                color = STAGE_IDLE_COLOR
            elif i < active_idx:
                color = STAGE_DONE_COLOR
            elif i == active_idx:
                color = STAGE_ACTIVE_COLOR
            else:
                color = STAGE_IDLE_COLOR
            self.stage_pills[name].configure(fg_color=color)

    def refresh(self) -> None:
        if self.closed:
            return
        d = self.controller.snapshot()

        if d.vis_rgb is not None:
            self._draw_camera(d.vis_rgb)

        self._draw_plot(d)
        self._sync_seg_panel(d)
        self._sync_click_panel()
        self._sync_uncertainty_panel(d)
        self.click_status_label.configure(text=d.click_status)
        self.mission_status_label.configure(text=d.mission_status)

        # REACH P2: drive the three-stage lifecycle pills.
        self._sync_stage_pills(self._derive_stage(d))

        mode_txt = d.mode.upper().replace("_", " ")
        if d.goal_reached:
            mode_txt += "  ✓ GOAL REACHED"
        self.mode_label.configure(
            text=mode_txt, text_color=MODE_COLORS.get(d.mode, "#e5e7eb")
        )
        # REACH P9: prefer the freshest human ack, fall back to controller
        # status text -- so the header reacts on the same frame as a press.
        self.detail_label.configure(text=self._ack_text or d.status_text)

        pose_txt = (
            f"pose ({d.pose.x:.1f}, {d.pose.z:.1f}, yaw={math.degrees(d.pose.yaw):.0f}deg)"
            if d.pose is not None
            else "pose: -"
        )
        act = d.action
        cbf_txt = ""
        if d.cbf_info.get("blocked"):
            cbf_txt = "  CBF:orbit" if d.cbf_info.get("orbiting") else "  CBF:blocked"
        if d.cbf_info.get("hard_gate_fired"):
            cbf_txt += "  CBF:hard-gate"
        unc_txt = (
            f"   unc={d.uncertainty_value:.2f}/{d.uncertainty_threshold:.2f}"
            if d.uncertainty_enabled
            else ""
        )
        self.telemetry_label.configure(
            text=f"{pose_txt}   step={d.step}   "
            f"v=[{act.v_fwd:.2f},{act.v_lat:.2f}] yaw_rate={act.yaw_rate:+.2f}{cbf_txt}{unc_txt}"
        )

        alive = self.controller.is_alive()
        if not alive and not self._alive_label_visible:
            self.alive_label.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 8))
            self._alive_label_visible = True
        elif alive and self._alive_label_visible:
            self.alive_label.grid_forget()
            self._alive_label_visible = False

        self.root.after(REFRESH_MS, self.refresh)

    def _sync_seg_panel(self, d) -> None:
        in_review = d.mode == MODE_REVIEW_SEGMENTATION
        if in_review and not self._seg_panel_visible:
            self.seg_panel.grid(
                row=self._seg_row, column=0, sticky="ew", pady=(SECTION_GAP, 0)
            )
            self._seg_panel_visible = True
            self._scroll_sidebar_to_bottom()
        elif not in_review and self._seg_panel_visible:
            self.seg_panel.grid_forget()
            self._seg_panel_visible = False
            self._scroll_sidebar_to_top()
        if in_review:
            self.seg_desc_label.configure(text=d.status_text)

    def _sync_click_panel(self) -> None:
        pending = self._pending_click_norm is not None
        if pending and not self._click_panel_visible:
            self.click_panel.grid(
                row=self._click_row, column=0, sticky="ew", pady=(SECTION_GAP, 0)
            )
            self._click_panel_visible = True
            self._scroll_sidebar_to_bottom()
        elif not pending and self._click_panel_visible:
            self.click_panel.grid_forget()
            self._click_panel_visible = False
            self._scroll_sidebar_to_top()

    def _sync_uncertainty_panel(self, d) -> None:
        if not d.uncertainty_halted:
            if self._uncertainty_panel_visible:
                self.uncertainty_panel.grid_forget()
                self._uncertainty_panel_visible = False
                self._scroll_sidebar_to_top()
            return

        if not self._uncertainty_panel_visible:
            self.uncertainty_panel.grid(
                row=self._uncertainty_row, column=0, sticky="ew", pady=(SECTION_GAP, 0)
            )
            self._uncertainty_panel_visible = True
            self._scroll_sidebar_to_bottom()

        in_flight = d.uncertainty_request_in_flight
        button_state = "disabled" if in_flight else "normal"
        for btn in self.uncertainty_buttons.values():
            btn.configure(state=button_state)
        self.uncertainty_retry_button.configure(state=button_state)

        if in_flight:
            self.uncertainty_panel.configure(border_color="#f59e0b")
            self.uncertainty_title.configure(
                text="REQUESTING VLM SWEEP...", text_color="#f59e0b"
            )
        else:
            self.uncertainty_panel.configure(border_color="#38bdf8")
            self.uncertainty_title.configure(
                text="UNCERTAINTY HALT -- choose a heading", text_color="#38bdf8"
            )
        self.uncertainty_desc_label.configure(
            text=d.uncertainty_line or "Waiting for VLM sweep description..."
        )

    def _draw_camera(self, vis_rgb) -> None:
        # Letterbox the (possibly non-square) frame into whatever size the
        # hero canvas currently is, so the view actually scales with the
        # window instead of sitting at a fixed pixel size.
        cw = self.cam_canvas.winfo_width()
        ch = self.cam_canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        img = Image.fromarray(vis_rgb).convert("RGB")
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        img = img.resize((dw, dh))
        if self._pending_click_norm is not None:
            img = self._draw_pending_click(img, dw, dh)
        self._cam_photo = ImageTk.PhotoImage(img)
        x0, y0 = (cw - dw) // 2, (ch - dh) // 2
        self.cam_canvas.delete("frame")
        self.cam_canvas.create_image(
            x0, y0, anchor="nw", image=self._cam_photo, tags="frame"
        )
        self._cam_img_box = (x0, y0, x0 + dw, y0 + dh)

    def _draw_pending_click(self, img: Image.Image, dw: int, dh: int) -> Image.Image:
        # Not-yet-confirmed marker at the last click, cyan to read as distinct
        # from the gold confirmed-goal marker the controller reprojects into
        # vis_rgb every frame once a click is confirmed (rover_controller's
        # draw_point_marker).
        nx, ny = self._pending_click_norm
        x, y = nx * dw, ny * dh
        r = max(6, min(dw, dh) // 48)
        draw = ImageDraw.Draw(img)
        draw.line([(x - r, y), (x + r, y)], fill="#00e5ff", width=2)
        draw.line([(x, y - r), (x, y + r)], fill="#00e5ff", width=2)
        draw.ellipse([x - r, y - r, x + r, y + r], outline="#00e5ff", width=2)
        return img

    def _draw_plot(self, d) -> None:
        self.plot.delete("all")
        S, R = self._plot_size, self.PLOT_RANGE

        def to_px(
            forward, left
        ):  # body frame (fwd, left) -> canvas (origin at rover, facing up)
            return S / 2 - (left / R) * (S / 2), S - (forward / R) * S * 0.92 - 20

        self.plot.create_line(0, S - 20, S, S - 20, fill=PLOT_AXIS)
        self.plot.create_oval(
            S / 2 - 5, S - 25, S / 2 + 5, S - 15, fill=PLOT_ROVER, outline=""
        )  # rover

        if d.obstacle_point is not None:
            ox, oy = to_px(d.obstacle_point[0], d.obstacle_point[1])
            self.plot.create_oval(
                ox - 5, oy - 5, ox + 5, oy + 5, outline=PLOT_OBSTACLE, width=2
            )

        if d.trajectory is not None and len(d.trajectory) > 1:
            pts = [to_px(float(p[0]), float(p[1])) for p in d.trajectory]
            self.plot.create_line(
                *[c for xy in pts for c in xy], fill=PLOT_TRAJECTORY, width=3
            )

        if d.belief_g is not None:
            gx, gy = to_px(d.belief_g[0], d.belief_g[1])
            self.plot.create_text(
                gx, gy, text="*", fill=PLOT_GOAL, font=("TkDefaultFont", 26)
            )

    def tick_forever(self) -> None:
        self.root.mainloop()


def build_controller(args: argparse.Namespace) -> RoverController:
    navdp_upstream_ckpt = args.navdp_upstream_ckpt or _default_navdp_upstream_ckpt()
    if not navdp_upstream_ckpt:
        raise ValueError(
            "--navdp-upstream-ckpt is required (no checkpoint found at the default "
            "navdp/navdp-cross-modal.ckpt either)"
        )
    return RoverController(
        scene_path=args.scene_path,
        heightmap_path=args.heightmap_path,
        navdp_upstream_ckpt=navdp_upstream_ckpt,
        navdp_upstream_root=args.navdp_upstream_root,
        navdp_root=args.navdp_root,
        rock_field_path=args.rock_field,
        flag_seed=args.flag_seed,
        num_flags=args.num_flags,
        flag_min_spacing=args.flag_min_spacing,
        flag_boundary_margin=args.flag_boundary_margin,
        start_x=args.start_x,
        start_z=args.start_z,
        start_yaw_deg=args.start_yaw,
        dt=args.dt,
        hz=args.hz,
        cbf_enabled=not args.no_cbf,
        cbf_d_safe=args.cbf_d_safe,
        cbf_gamma=args.cbf_gamma,
        cbf_deadzone=args.cbf_deadzone,
        max_forward_speed=args.max_linear,
        max_yaw_rate=args.max_angular,
        navdp_upstream_port=args.navdp_upstream_port,
        navdp_upstream_lookahead=args.navdp_upstream_lookahead,
        navdp_upstream_replan_every=args.navdp_upstream_replan_every,
        navdp_upstream_server_variant=args.navdp_upstream_server_variant,
        navdp_upstream_planner_mode=args.navdp_upstream_planner_mode,
        random_goal_bearing_deg=args.random_goal_bearing_deg,
        random_goal_dist_range=(args.random_goal_min_dist, args.random_goal_max_dist),
        seg_backend=args.seg_backend,
        seg_checkpoint=args.seg_checkpoint,
        seg_overlay=args.seg_overlay,
        annotations_dir=args.annotations_dir,
        annotation_categories=args.annotation_categories,
        dino_model_id=args.dino_model_id,
        dino_device=args.dino_device,
        dino_box_threshold=args.dino_box_threshold,
        dino_text_threshold=args.dino_text_threshold,
        uncertainty_enabled=not args.no_uncertainty_halt,
        uncertainty_cov_threshold=args.cov_threshold,
        uncertainty_cov_growth=args.cov_growth,
        uncertainty_cov_growth_rate=args.cov_growth_rate,
        uncertainty_search_dist=args.uncertainty_search_dist,
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--scene-path", default=str(REPO_ROOT / "assets" / "marsyard2022.glb")
    )
    ap.add_argument(
        "--heightmap-path", default=str(REPO_ROOT / "marsyard2022_terrain_hm_1025.tif")
    )
    ap.add_argument(
        "--rock-field", default=None, help="rock_field.json manifest (optional)"
    )
    ap.add_argument(
        "--flag-seed",
        type=int,
        default=7,  # 1
        help="enable randomized flag-marker placement (assets/flags/*.glb) seeded with "
        "this value -- same seed gives the same layout every run; unset (default) "
        "places no flags",
    )
    ap.add_argument(
        "--num-flags",
        type=int,
        default=6,
        help="flags to place when --flag-seed is set",
    )
    ap.add_argument(
        "--flag-min-spacing",
        type=float,
        default=1.5,
        help="minimum meters between placed flags",
    )
    ap.add_argument(
        "--flag-boundary-margin",
        type=float,
        default=2.0,
        help="keep placed flags this many meters clear of the scene bounds",
    )
    ap.add_argument(
        "--navdp-upstream-ckpt",
        default=None,
        help="Path to the upstream NavDP .ckpt (default: navdp/navdp-cross-modal.ckpt if present)",
    )
    ap.add_argument(
        "--navdp-upstream-root",
        default=None,
        help="Path to the vendored InternRobotics/NavDP checkout (default: $NAVDP_UPSTREAM_ROOT)",
    )
    ap.add_argument("--navdp-upstream-port", type=int, default=None)
    ap.add_argument("--navdp-upstream-lookahead", type=int, default=3)
    ap.add_argument("--navdp-upstream-replan-every", type=int, default=1)
    ap.add_argument(
        "--navdp-upstream-server-variant",
        choices=["navdp", "s2diff"],
        default="navdp",
        help="'navdp' spawns InternRobotics/NavDP's own baselines/navdp/navdp_server.py; "
        "'s2diff' spawns this project's obstacle-guided navdp_s2diff_server.py fork instead "
        "(same checkpoint, see navdp_upstream_server_manager.py's docstring)",
    )
    ap.add_argument(
        "--navdp-upstream-planner-mode",
        choices=["pure-navdp", "s2diff", "gradient"],
        default="s2diff",
        help="server_variant=s2diff only: which guidance mode navdp_s2diff_server.py runs",
    )
    ap.add_argument(
        "--navdp-root",
        default=None,
        help="Path to this repo's own navdp/ package (default: ./navdp or $NAVDP_ROOT) -- only "
        "needed for the CBF safety layer's generic obstacle/avoidance math, unrelated to which "
        "driving policy is active",
    )
    ap.add_argument(
        "--start-x", type=float, default=-5.5
    )  # 7.1, 7.6, 2.2, 7.5, # 1.7, # -5.5
    ap.add_argument(
        "--start-z", type=float, default=2.2
    )  # 7.7, 7.1, -1.9, 6.9, # 0.7 # 2.2
    ap.add_argument(
        "--start-yaw", type=float, default=86.0, help="degrees"
    )  # 34, 41, 161, 34, # 58 # 86
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--hz", type=float, default=10.0, help="controller tick rate")
    ap.add_argument("--max-linear", type=float, default=0.6)
    ap.add_argument("--max-angular", type=float, default=0.6)
    ap.add_argument(
        "--no-cbf", action="store_true", help="disable CBF cone-mode obstacle avoidance"
    )
    ap.add_argument("--cbf-d-safe", type=float, default=0.75)
    ap.add_argument("--cbf-gamma", type=float, default=0.3)
    ap.add_argument("--cbf-deadzone", type=float, default=0.6)
    ap.add_argument("--random-goal-bearing-deg", type=float, default=60.0)
    ap.add_argument("--random-goal-min-dist", type=float, default=4.0)
    ap.add_argument("--random-goal-max-dist", type=float, default=8.0)
    ap.add_argument(
        "--seg-backend",
        choices=["lora", "legacy"],
        default="lora",
        help="segmentation checkpoint used by 'Segment' goal resolution: 'lora' (default) is "
        "sam_lora_runs/exp10/best, LoRA-finetuned on mesh_tight_bound2-overlay frames; "
        "'legacy' is the original single-checkpoint model (best_model.pth)",
    )
    ap.add_argument(
        "--seg-checkpoint",
        default=None,
        help="override the checkpoint path (legacy: a .pth file) or dir (lora: a "
        "finetune_sam2_lora.py out-dir, e.g. sam_lora_runs/exp10/best) for --seg-backend",
    )
    ap.add_argument(
        "--seg-overlay",
        choices=["mesh", "none"],
        default="mesh",
        help="'mesh' (default) composites --annotations-dir's hull meshes into a separate "
        "frame fed only to the segmentation model, matching what the default lora checkpoint "
        "was trained on -- never shown in the GUI or sent to the goal-selection VLM. "
        "'none' segments the plain camera frame",
    )
    ap.add_argument(
        "--annotations-dir",
        default=str(REPO_ROOT / "annotations" / "mesh_tight_bound2"),
        help="hull-mesh annotation dir used by --seg-overlay=mesh (default: "
        "annotations/mesh_tight_bound2, the dataset sam_lora_runs/exp10 was trained on)",
    )
    ap.add_argument(
        "--annotation-categories",
        nargs="+",
        default=None,
        help="restrict --seg-overlay=mesh to these hull categories (default: all)",
    )
    ap.add_argument(
        "--dino-model-id",
        default=dino_grounding_resolver.DEFAULT_DINO_MODEL_ID,
        help="GroundingDINO checkpoint for open-vocabulary target grounding "
        "('Ground Target' / a Command-panel GO_TO/FIND step) -- HF "
        "AutoModelForZeroShotObjectDetection id, lazily loaded on first use",
    )
    ap.add_argument(
        "--dino-device",
        default=dino_grounding_resolver.DEFAULT_DINO_DEVICE,
        help="device GroundingDINO loads onto (default cuda)",
    )
    ap.add_argument(
        "--dino-box-threshold",
        type=float,
        default=dino_grounding_resolver.DEFAULT_BOX_THRESHOLD,
    )
    ap.add_argument(
        "--dino-text-threshold",
        type=float,
        default=dino_grounding_resolver.DEFAULT_TEXT_THRESHOLD,
    )
    ap.add_argument(
        "--no-uncertainty-halt",
        action="store_true",
        help="disable the uncertainty-halt heading-request prompt (vl_direction's "
        "'uncertainty' mode against the resolved goal's real BeliefGoalTracker -- see "
        "--cov-threshold/--cov-growth). Default on; only spins up its own Qwen VLM "
        "subprocess (distinct from --seg-backend's) once this halt actually fires",
    )
    ap.add_argument(
        "--cov-threshold",
        type=float,
        default=vl_dir_config.DEFAULT_COVARIANCE_THRESHOLD,
        help="belief-uncertainty value (BeliefGoalTracker.uncertainty_value(), grows "
        "while a resolved goal mask stays unseen) at which MODE_RESOLVE driving halts "
        f"and a heading is requested (default {vl_dir_config.DEFAULT_COVARIANCE_THRESHOLD})",
    )
    ap.add_argument(
        "--cov-growth",
        type=float,
        default=0.01,
        help="base per-tick belief-uncertainty growth while the goal mask is unseen "
        "(default 0.01); lower this to trigger the halt sooner for testing",
    )
    ap.add_argument(
        "--cov-growth-rate",
        type=float,
        default=0.0,
        help="accelerating-drift factor: growth speeds up the longer the goal has been "
        "unseen (default 0.0, i.e. flat per-tick growth unless overridden)",
    )
    ap.add_argument(
        "--uncertainty-search-dist",
        type=float,
        default=4.0,
        help="meters driven along a human-submitted heading before falling back to the "
        "(still-uncertain) dead-reckoned belief and, if still unseen, halting again "
        "(default 4.0)",
    )
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    controller = build_controller(args)
    controller.start()

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    root = ctk.CTk()
    app = NavGuiApp(
        root, controller, max_linear=args.max_linear, max_angular=args.max_angular
    )
    try:
        app.tick_forever()
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
