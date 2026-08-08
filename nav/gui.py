#!/usr/bin/env python3
"""Interactive control panel for nav.rover_controller.RoverController --
the in-house, single-process equivalent of Nav_new/MARS/launch_mars.sh's
DINO+NavDP GUI, driving the actual published NavDP model instead of this
repo's own custom S2DiT+NavDP model (see rover_controller.py's docstring).

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
from tkinter import ttk
from PIL import Image, ImageTk

from nav.rover_controller import RoverController

REPO_ROOT = Path(__file__).resolve().parent.parent
REFRESH_MS = 66  # ~15 Hz GUI repaint, independent of the controller's own hz


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

    def __init__(self, root: tk.Tk, controller: RoverController, max_linear: float, max_angular: float):
        self.root = root
        self.controller = controller
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.closed = False
        self._manual_held: set[str] = set()

        root.title("mars-habitatsim/nav -- NavDP (upstream) rover control")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        main = ttk.Frame(root, padding=8)
        main.grid(sticky="nsew")

        self.cam_label = ttk.Label(main)
        self.cam_label.grid(row=0, column=0, padx=4, pady=4)
        self._blank_photo = ImageTk.PhotoImage(Image.new("RGB", (self.CAM_SIZE, self.CAM_SIZE), "#222"))
        self.cam_label.configure(image=self._blank_photo)

        self.plot = tk.Canvas(main, width=self.PLOT_SIZE, height=self.PLOT_SIZE, bg="white")
        self.plot.grid(row=0, column=1, padx=4, pady=4)

        bar = ttk.Frame(main)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        ttk.Button(bar, text="Resolve Goal (auto)", command=self.resolve_goal).pack(side="left", padx=2)
        ttk.Button(bar, text="Random Goal", command=self.random_goal).pack(side="left", padx=8)
        ttk.Button(bar, text="Go Home", command=self.go_home).pack(side="left", padx=2)
        ttk.Button(bar, text="Reset Rover", command=self.reset_rover).pack(side="left", padx=8)
        ttk.Button(bar, text="STOP", command=self.stop).pack(side="left", padx=10)

        drive = ttk.Frame(main)
        drive.grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(drive, text="Manual drive (hold, or arrow keys):").pack(side="left")
        for label, direction in (("<", "left"), ("^", "fwd"), ("v", "back"), (">", "right")):
            b = ttk.Button(drive, text=label, width=3)
            b.bind("<ButtonPress-1>", lambda e, d=direction: self.manual_press(d))
            b.bind("<ButtonRelease-1>", lambda e, d=direction: self.manual_release(d))
            b.pack(side="left", padx=2)
        for key, direction in (("Up", "fwd"), ("Down", "back"), ("Left", "left"), ("Right", "right")):
            root.bind(f"<KeyPress-{key}>", lambda e, d=direction: self.manual_press(d))
            root.bind(f"<KeyRelease-{key}>", lambda e, d=direction: self.manual_release(d))

        self.status = ttk.Label(
            main, text="starting...", font=("TkDefaultFont", 11, "bold"), width=110, anchor="w"
        )
        self.status.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.info = ttk.Label(main, text="", width=110, anchor="w")
        self.info.grid(row=4, column=0, columnspan=2, sticky="w")

        self._photo = None
        self.root.after(REFRESH_MS, self.refresh)

    # ---------------- commands ---------------- #
    def resolve_goal(self) -> None:
        self._manual_held.clear()
        self.controller.request_resolve()

    def random_goal(self) -> None:
        self._manual_held.clear()
        self.controller.random_goal()

    def go_home(self) -> None:
        self._manual_held.clear()
        self.controller.go_home()

    def reset_rover(self) -> None:
        self._manual_held.clear()
        self.controller.request_reset()

    def stop(self) -> None:
        self._manual_held.clear()
        self.controller.stop_driving()

    # ---------------- manual drive ---------------- #
    def manual_press(self, direction: str) -> None:
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
            self._photo = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=self._photo)

        self._draw_plot(d)

        mode_txt = d.mode.upper()
        if d.goal_reached:
            mode_txt += " (GOAL REACHED)"
        alive_txt = "" if self.controller.is_alive() else "  [controller thread died -- see console]"
        self.status.configure(text=f"[{mode_txt}] {d.status_text}{alive_txt}")

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
        self.info.configure(
            text=f"{pose_txt}   step={d.step}  frames={d.frame_count}   "
            f"v=[{act.v_fwd:.2f},{act.v_lat:.2f}] yaw_rate={act.yaw_rate:+.2f}{cbf_txt}"
        )

        self.root.after(REFRESH_MS, self.refresh)

    def _draw_plot(self, d) -> None:
        self.plot.delete("all")
        S, R = self.PLOT_SIZE, self.PLOT_RANGE

        def to_px(forward, left):  # body frame (fwd, left) -> canvas (origin at rover, facing up)
            return S / 2 - (left / R) * (S / 2), S - (forward / R) * S * 0.92 - 20

        self.plot.create_line(0, S - 20, S, S - 20, fill="#ddd")
        self.plot.create_oval(S / 2 - 5, S - 25, S / 2 + 5, S - 15, fill="black")  # rover

        if d.obstacle_point is not None:
            ox, oy = to_px(d.obstacle_point[0], d.obstacle_point[1])
            self.plot.create_oval(ox - 5, oy - 5, ox + 5, oy + 5, outline="#c0392b", width=2)

        if d.trajectory is not None and len(d.trajectory) > 1:
            pts = [to_px(float(p[0]), float(p[1])) for p in d.trajectory]
            self.plot.create_line(*[c for xy in pts for c in xy], fill="red", width=3)

        if d.belief_g is not None:
            gx, gy = to_px(d.belief_g[0], d.belief_g[1])
            self.plot.create_text(gx, gy, text="*", fill="#d4a017", font=("TkDefaultFont", 26))

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
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-path", default=str(REPO_ROOT / "assets" / "marsyard2022.glb"))
    ap.add_argument("--heightmap-path", default=str(REPO_ROOT / "marsyard2022_terrain_hm_1025.tif"))
    ap.add_argument("--rock-field", default=None, help="rock_field.json manifest (optional)")
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
    ap.add_argument("--start-x", type=float, default=0.0)
    ap.add_argument("--start-z", type=float, default=8.0)
    ap.add_argument("--start-yaw", type=float, default=0.0, help="degrees")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--hz", type=float, default=10.0, help="controller tick rate")
    ap.add_argument("--max-linear", type=float, default=0.6)
    ap.add_argument("--max-angular", type=float, default=0.6)
    ap.add_argument("--no-cbf", action="store_true", help="disable CBF cone-mode obstacle avoidance")
    ap.add_argument("--cbf-d-safe", type=float, default=0.75)
    ap.add_argument("--cbf-gamma", type=float, default=0.3)
    ap.add_argument("--cbf-deadzone", type=float, default=0.6)
    ap.add_argument("--random-goal-bearing-deg", type=float, default=60.0)
    ap.add_argument("--random-goal-min-dist", type=float, default=4.0)
    ap.add_argument("--random-goal-max-dist", type=float, default=8.0)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    controller = build_controller(args)
    controller.start()

    root = tk.Tk()
    app = NavGuiApp(root, controller, max_linear=args.max_linear, max_angular=args.max_angular)
    try:
        app.tick_forever()
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
