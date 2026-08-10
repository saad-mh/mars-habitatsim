#!/usr/bin/env python3
"""Interactive control panel for nav.rover_controller.RoverController --
the in-house, single-process equivalent of Nav_new/MARS/launch_mars.sh's
DINO+NavDP GUI, driving the actual published NavDP model instead of this
repo's own custom S2DiT+NavDP model (see rover_controller.py's docstring).

UI follows the same customtkinter conventions as
scripts/habitat_tests/kb_teleop_vl.py: dark theme, grouped CTkFrame panels,
and state-only panels (segmentation review, click-to-goal confirm) that
appear/disappear via pack()/pack_forget() instead of sitting on screen
disabled -- mirrors that script's uncertainty-halt panel.

Run via nav/launch_nav.sh, or directly:
    conda activate habitat
    cd mars-habitatsim
    python -m nav.gui [--scene-path ...] [--heightmap-path ...] [--cbf/--no-cbf] ...
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk

from nav.rover_controller import MODE_REVIEW_SEGMENTATION, RoverController

REPO_ROOT = Path(__file__).resolve().parent.parent
REFRESH_MS = 66  # ~15 Hz GUI repaint, independent of the controller's own hz

PLOT_BG = "#242424"
PLOT_AXIS = "#555555"
PLOT_ROVER = "#e5e7eb"
PLOT_OBSTACLE = "#f87171"
PLOT_TRAJECTORY = "#f97373"
PLOT_GOAL = "#fbbf24"

# Mode -> accent color, shared by the big mode label and (where relevant)
# the panel that goes with that mode -- keeps color the one consistent cue
# for "what is the rover doing right now" across the whole window.
MODE_COLORS = {
    "idle": "#9ca3af",
    "manual": "#60a5fa",
    "point": "#4ade80",
    "resolve": "#c084fc",
    MODE_REVIEW_SEGMENTATION: "#f59e0b",
}


def _default_navdp_upstream_ckpt() -> Optional[str]:
    """Same repo-relative fallback sam_vla.run_navdp_rollout uses -- a plain
    invocation just works on a checkout that already has Phase 0's
    (next.md's Integration project) gitignored checkpoint in place."""
    candidate = REPO_ROOT / "navdp" / "navdp-cross-modal.ckpt"
    return str(candidate) if candidate.exists() else None


class NavGuiApp:
    CAM_SIZE = 480
    PLOT_SIZE = 420
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

        root.title("mars-habitatsim/nav")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # The packed content (camera/plot row + all panels, with both
        # state-only panels visible at once) is taller than a lot of
        # monitors -- cap the window to the screen and let this scrollable
        # frame carry the overflow instead of the window growing off-screen
        # with no way to reach the bottom row of buttons.
        screen_h = root.winfo_screenheight()
        window_h = min(int(screen_h * 0.9), 1000)
        root.geometry(f"1000x{window_h}")
        root.minsize(760, 480)

        content = ctk.CTkScrollableFrame(root, fg_color="transparent")
        content.pack(fill="both", expand=True)

        mono_font = ctk.CTkFont(family="Consolas", size=12)
        mode_font = ctk.CTkFont(size=17, weight="bold")

        # -- camera view + body-frame plot, side by side -- #
        top_row = ctk.CTkFrame(content, fg_color="transparent")
        top_row.pack(padx=10, pady=(10, 4))

        cam_frame = ctk.CTkFrame(top_row)
        cam_frame.pack(side="left", padx=(0, 4))
        # Plain tk.Label (not CTkLabel) for the per-frame video blit, same
        # reasoning as kb_teleop_vl.py: a raw PhotoImage swap every refresh
        # is cheaper than rebuilding a CTkImage each time.
        self.cam_label = tk.Label(cam_frame, bd=0)
        self.cam_label.pack(padx=4, pady=(4, 2))
        self._blank_photo = ImageTk.PhotoImage(
            Image.new("RGB", (self.CAM_SIZE, self.CAM_SIZE), "#222")
        )
        self.cam_label.configure(image=self._blank_photo)
        self.cam_label.bind("<Button-1>", self.on_cam_click)
        ctk.CTkLabel(
            cam_frame,
            text="click the view to set a goal point",
            font=ctk.CTkFont(size=11),
            text_color="#9ca3af",
        ).pack(pady=(0, 4))

        plot_frame = ctk.CTkFrame(top_row)
        plot_frame.pack(side="left", padx=(4, 0))
        ctk.CTkLabel(
            plot_frame, text="body-frame view", font=ctk.CTkFont(size=11)
        ).pack(pady=(4, 0))
        self.plot = tk.Canvas(
            plot_frame,
            width=self.PLOT_SIZE,
            height=self.PLOT_SIZE,
            bg=PLOT_BG,
            highlightthickness=0,
        )
        self.plot.pack(padx=4, pady=(2, 4))

        # -- status panel: mode + detail + telemetry, always visible -- #
        self.status_panel = ctk.CTkFrame(content)
        self.status_panel.pack(fill="x", padx=10, pady=4)
        self.mode_label = ctk.CTkLabel(
            self.status_panel, text="", anchor="w", justify="left", font=mode_font
        )
        self.mode_label.pack(fill="x", padx=10, pady=(8, 0))
        self.detail_label = ctk.CTkLabel(
            self.status_panel,
            text="",
            anchor="w",
            justify="left",
            wraplength=860,
            font=ctk.CTkFont(size=12),
        )
        self.detail_label.pack(fill="x", padx=10, pady=(2, 2))
        self.telemetry_label = ctk.CTkLabel(
            self.status_panel,
            text="",
            anchor="w",
            justify="left",
            font=mono_font,
            text_color="#9ca3af",
        )
        self.telemetry_label.pack(fill="x", padx=10, pady=(0, 2))
        self.alive_label = ctk.CTkLabel(
            self.status_panel,
            text="controller thread died -- see console",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#f87171",
        )
        self._alive_label_visible = False

        # -- primary action bar -- #
        actions = ctk.CTkFrame(content)
        actions.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(
            actions, text="Goal", font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=(10, 10), pady=8)
        ctk.CTkButton(
            actions, text="Segment", command=self.resolve_goal, width=150
        ).grid(row=0, column=1, padx=4, pady=8)
        ctk.CTkButton(
            actions, text="Random Goal", command=self.random_goal, width=120
        ).grid(row=0, column=2, padx=4, pady=8)
        ctk.CTkButton(actions, text="Go Home", command=self.go_home, width=100).grid(
            row=0, column=3, padx=4, pady=8
        )
        ctk.CTkButton(
            actions,
            text="Reset Rover",
            command=self.reset_rover,
            width=110,
            fg_color="#4b5563",
            hover_color="#374151",
        ).grid(row=0, column=4, padx=(20, 4), pady=8)
        ctk.CTkButton(
            actions,
            text="STOP",
            command=self.stop,
            width=90,
            fg_color="#b91c1c",
            hover_color="#991b1b",
        ).grid(row=0, column=5, padx=(4, 10), pady=8)

        # -- manual drive: D-pad, always visible -- #
        drive = ctk.CTkFrame(content)
        drive.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(
            drive,
            text="Manual Drive",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, rowspan=3, sticky="w", padx=(10, 16), pady=8)
        dpad = ctk.CTkFrame(drive, fg_color="transparent")
        dpad.grid(row=0, column=1, rowspan=3, pady=6)
        dpad_cells = {
            "fwd": (0, 1, "↑"),
            "left": (1, 0, "←"),
            "right": (1, 2, "→"),
            "back": (2, 1, "↓"),
        }
        for direction, (r, c, glyph) in dpad_cells.items():
            b = ctk.CTkButton(dpad, text=glyph, width=56, height=44)
            b.bind("<ButtonPress-1>", lambda e, d=direction: self.manual_press(d))
            b.bind("<ButtonRelease-1>", lambda e, d=direction: self.manual_release(d))
            b.grid(row=r, column=c, padx=4, pady=4)
        ctk.CTkLabel(
            drive,
            text="hold a direction, or use the arrow keys -- Esc cancels a pending click",
            font=ctk.CTkFont(size=11),
            text_color="#9ca3af",
        ).grid(row=0, column=2, sticky="w", padx=10)

        for key, direction in (
            ("Up", "fwd"),
            ("Down", "back"),
            ("Left", "left"),
            ("Right", "right"),
        ):
            root.bind(f"<KeyPress-{key}>", lambda e, d=direction: self.manual_press(d))
            root.bind(
                f"<KeyRelease-{key}>", lambda e, d=direction: self.manual_release(d)
            )
        root.bind("<Escape>", lambda e: self.cancel_pixel_click())

        # -- segmentation review panel: only packed while actually
        # reviewing (Goal 1) -- bordered like kb_teleop_vl's uncertainty
        # halt panel so it reads as "action needed", not a fourth static
        # button row. -- #
        self.seg_panel = ctk.CTkFrame(content, border_width=2, border_color="#f59e0b")
        ctk.CTkLabel(
            self.seg_panel,
            text="REVIEWING RESOLVED GOAL",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#f59e0b",
        ).pack(pady=(10, 2))
        self.seg_desc_label = ctk.CTkLabel(
            self.seg_panel, text="", wraplength=860, justify="left"
        )
        self.seg_desc_label.pack(padx=12, pady=(0, 8), fill="x")
        seg_btn_row = ctk.CTkFrame(self.seg_panel, fg_color="transparent")
        seg_btn_row.pack(pady=(0, 12))
        ctk.CTkButton(
            seg_btn_row,
            text="Confirm",
            command=self.confirm_segmentation,
            width=120,
            fg_color="#15803d",
            hover_color="#166534",
        ).grid(row=0, column=0, padx=6)
        ctk.CTkButton(
            seg_btn_row,
            text="Rerun",
            command=self.rerun_segmentation,
            width=120,
            fg_color="#b45309",
            hover_color="#92400e",
        ).grid(row=0, column=1, padx=6)
        ctk.CTkButton(
            seg_btn_row,
            text="Pick Manually",
            command=self.pick_manually,
            width=140,
        ).grid(row=0, column=2, padx=6)

        # -- click-to-goal confirm panel: only packed while a click is
        # pending confirmation. -- #
        self.click_panel = ctk.CTkFrame(content, border_width=2, border_color="#00b8d4")
        self.click_desc_label = ctk.CTkLabel(
            self.click_panel,
            text="Set the goal at the point you clicked?",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#22d3ee",
        )
        self.click_desc_label.pack(pady=(10, 6))
        click_btn_row = ctk.CTkFrame(self.click_panel, fg_color="transparent")
        click_btn_row.pack(pady=(0, 10))
        ctk.CTkButton(
            click_btn_row,
            text="Confirm Point Goal",
            command=self.confirm_pixel_goal,
            width=160,
            fg_color="#15803d",
            hover_color="#166534",
        ).grid(row=0, column=0, padx=6)
        ctk.CTkButton(
            click_btn_row,
            text="Cancel",
            command=self.cancel_pixel_click,
            width=100,
            fg_color="#4b5563",
            hover_color="#374151",
        ).grid(row=0, column=1, padx=6)

        # -- persistent click-result line (last click's outcome, e.g. "point
        # goal set at world (x, z)" or "click ignored: no valid depth
        # there") -- stays visible after the confirm panel above closes. -- #
        self.click_status_label = ctk.CTkLabel(
            content,
            text="",
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color="#9ca3af",
        )
        self.click_status_label.pack(fill="x", padx=14, pady=(0, 8))

        self._photo = None
        self.root.after(REFRESH_MS, self.refresh)

    # ---------------- commands ---------------- #
    def resolve_goal(self) -> None:
        self._manual_held.clear()
        self.cancel_pixel_click()
        self.controller.request_resolve()

    def random_goal(self) -> None:
        self._manual_held.clear()
        self.cancel_pixel_click()
        self.controller.random_goal()

    def go_home(self) -> None:
        self._manual_held.clear()
        self.cancel_pixel_click()
        self.controller.go_home()

    def reset_rover(self) -> None:
        self._manual_held.clear()
        self.cancel_pixel_click()
        self.controller.request_reset()

    def stop(self) -> None:
        self._manual_held.clear()
        self.cancel_pixel_click()
        self.controller.stop_driving()

    # ---------------- segmentation review (Goal 1) ---------------- #
    def confirm_segmentation(self) -> None:
        self.controller.request_confirm_segmentation()

    def rerun_segmentation(self) -> None:
        self.controller.request_rerun_segmentation()

    def pick_manually(self) -> None:
        self._manual_held.clear()
        self.controller.request_pick_manually()

    # ---------------- click-to-goal ---------------- #
    def on_cam_click(self, event) -> None:
        nx = min(max(event.x / self.CAM_SIZE, 0.0), 1.0)
        ny = min(max(event.y / self.CAM_SIZE, 0.0), 1.0)
        self._pending_click_norm = (nx, ny)

    def confirm_pixel_goal(self) -> None:
        if self._pending_click_norm is None:
            return
        self._manual_held.clear()
        nx, ny = self._pending_click_norm
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
    def refresh(self) -> None:
        if self.closed:
            return
        d = self.controller.snapshot()

        if d.vis_rgb is not None:
            img = Image.fromarray(d.vis_rgb).convert("RGB")
            img = img.resize((self.CAM_SIZE, self.CAM_SIZE))
            if self._pending_click_norm is not None:
                img = self._draw_pending_click(img)
            self._photo = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=self._photo)

        self._draw_plot(d)
        self._sync_seg_panel(d)
        self._sync_click_panel()
        self.click_status_label.configure(text=d.click_status)

        mode_txt = d.mode.upper().replace("_", " ")
        if d.goal_reached:
            mode_txt += " "  # Goal Reached!
        self.mode_label.configure(
            text=mode_txt, text_color=MODE_COLORS.get(d.mode, "#e5e7eb")
        )
        self.detail_label.configure(text=d.status_text)

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
        self.telemetry_label.configure(
            text=f"{pose_txt}   step={d.step}   "
            f"v=[{act.v_fwd:.2f},{act.v_lat:.2f}] yaw_rate={act.yaw_rate:+.2f}{cbf_txt}"
        )

        alive = self.controller.is_alive()
        if not alive and not self._alive_label_visible:
            self.alive_label.pack(fill="x", padx=10, pady=(0, 8))
            self._alive_label_visible = True
        elif alive and self._alive_label_visible:
            self.alive_label.pack_forget()
            self._alive_label_visible = False

        self.root.after(REFRESH_MS, self.refresh)

    def _sync_seg_panel(self, d) -> None:
        in_review = d.mode == MODE_REVIEW_SEGMENTATION
        if in_review and not self._seg_panel_visible:
            self.seg_panel.pack(fill="x", padx=10, pady=4)
            self._seg_panel_visible = True
        elif not in_review and self._seg_panel_visible:
            self.seg_panel.pack_forget()
            self._seg_panel_visible = False
        if in_review:
            self.seg_desc_label.configure(text=d.status_text)

    def _sync_click_panel(self) -> None:
        pending = self._pending_click_norm is not None
        if pending and not self._click_panel_visible:
            self.click_panel.pack(fill="x", padx=10, pady=4)
            self._click_panel_visible = True
        elif not pending and self._click_panel_visible:
            self.click_panel.pack_forget()
            self._click_panel_visible = False

    def _draw_pending_click(self, img: Image.Image) -> Image.Image:
        # Not-yet-confirmed marker at the last click, cyan to read as distinct
        # from the gold confirmed-goal marker the controller reprojects into
        # vis_rgb every frame once a click is confirmed (rover_controller's
        # draw_point_marker).
        nx, ny = self._pending_click_norm
        x, y = nx * self.CAM_SIZE, ny * self.CAM_SIZE
        r = 10
        draw = ImageDraw.Draw(img)
        draw.line([(x - r, y), (x + r, y)], fill="#00e5ff", width=2)
        draw.line([(x, y - r), (x, y + r)], fill="#00e5ff", width=2)
        draw.ellipse([x - r, y - r, x + r, y + r], outline="#00e5ff", width=2)
        return img

    def _draw_plot(self, d) -> None:
        self.plot.delete("all")
        S, R = self.PLOT_SIZE, self.PLOT_RANGE

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
        random_goal_bearing_deg=args.random_goal_bearing_deg,
        random_goal_dist_range=(args.random_goal_min_dist, args.random_goal_max_dist),
        seg_backend=args.seg_backend,
        seg_checkpoint=args.seg_checkpoint,
        seg_overlay=args.seg_overlay,
        annotations_dir=args.annotations_dir,
        annotation_categories=args.annotation_categories,
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
        "--navdp-root",
        default=None,
        help="Path to this repo's own navdp/ package (default: ./navdp or $NAVDP_ROOT) -- only "
        "needed for the CBF safety layer's generic obstacle/avoidance math, unrelated to which "
        "driving policy is active",
    )
    ap.add_argument("--start-x", type=float, default=8.0)  # 7.1
    ap.add_argument("--start-z", type=float, default=10.0)  # 7.7
    ap.add_argument("--start-yaw", type=float, default=0.0, help="degrees")  # 34
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
