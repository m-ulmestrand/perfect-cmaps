import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perfect_cmaps.color_utils import (
    get_lightness_profile,
    rgb_renormalized_lightness,
    interpolate_lab,
    Lab_to_sRGB,
    SUPPORTED_L_PROFILES
)
from perfect_cmaps.gamut_view import (
    LAB_EXTENT,
    GamutBoundary,
    ab_to_square,
    clamp_to_gamut,
    gamut_boundary,
    relative_chroma,
    render_lab,
    render_square,
    square_to_ab,
    square_to_ab_exact,
)
from perfect_cmaps.storage import save_data
from perfect_cmaps.lightness_fit import DEFAULT_INTERPOLATION, ensure_fit


VIEWS = ("lab", "square")

# Dark, neutral theme so that the colors stand out. Applied with plt.rc_context,
# so the user's own matplotlib style is left alone.
BACKGROUND = "#1c1c1e"
PANEL = "#2a2a2d"
EDGE = "#4a4a4f"
TEXT = "#e6e6e6"
MUTED = "#8e8e93"
ACCENT = "#ffffff"
PLACEHOLDER = "#6a6a70"
PANEL_RGB = [int(PANEL[i:i + 2], 16) / 255 for i in (1, 3, 5)]



def _available_fonts(preferred: List[str], fallback: str) -> List[str]:
    """Filter a font preference list to the installed fonts, so matplotlib doesn't warn."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return [name for name in preferred if name in installed] + [fallback]


SANS_FONTS = _available_fonts(
    ["Inter", "SF Pro Text", "Helvetica Neue", "Segoe UI", "Roboto", "Ubuntu", "Noto Sans",
     "Open Sans", "Source Sans 3", "Source Sans Pro", "Arial", "Liberation Sans"],
    "DejaVu Sans",
)
MONO_FONTS = _available_fonts(
    ["JetBrains Mono", "SF Mono", "Menlo", "Cascadia Mono", "Consolas", "Fira Code", "Fira Mono",
     "Roboto Mono", "Source Code Pro", "Ubuntu Mono", "Noto Sans Mono", "Liberation Mono"],
    "DejaVu Sans Mono",
)

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": SANS_FONTS,
    "font.monospace": MONO_FONTS,
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": PANEL,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12.5,
    "axes.titleweight": "semibold",
    "axes.titlepad": 8,
    "axes.labelsize": 13,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 9,
    "toolbar": "None",
    # Free keys that the creator uses from matplotlib's default navigation shortcuts
    "keymap.back": ["left", "MouseButton.BACK"],
    "keymap.forward": ["right", "MouseButton.FORWARD"],
    "keymap.quit": ["ctrl+w", "cmd+w"],
}

PREVIEW_SAMPLES = 1000
FINAL_SMOOTHING = 0.03
HELP_TEXT = "Click: choose color   ·   Right-click / Backspace: undo   ·   V: switch view   ·   Esc: quit"


def _lab_to_display_rgb(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lab = np.column_stack((np.atleast_1d(L), np.atleast_1d(a), np.atleast_1d(b)))
    return np.clip(Lab_to_sRGB(lab), 0.0, 1.0)


def _hex(rgb: np.ndarray) -> str:
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)


class ColormapCreator:
    """Interactive figure for choosing one (a*, b*) control point per L* level."""

    def __init__(
            self,
            num_points: int = 20,
            lightness: str = "linear",
            view: str = "lab",
            num_bins: int = 500,
            default_name: str = "custom_cmap",
        ):
        assert lightness in SUPPORTED_L_PROFILES, \
            f"Lightness profile not in supported profiles. Valid choices are {SUPPORTED_L_PROFILES}"
        assert view in VIEWS, f"view must be one of {VIEWS}"
        assert num_points >= 2, "At least two control points are needed"

        self.lightness = lightness
        self.view = view
        self.num_bins = num_bins
        self.L_values = get_lightness_profile(num_points, lightness)
        self.points: List[Tuple[float, float]] = []
        self.saved_name: Optional[str] = None
        self.final_rgb: Optional[np.ndarray] = None

        self._boundaries: Dict[int, GamutBoundary] = {}
        self._images: Dict[Tuple[int, str], np.ndarray] = {}

        self._build_figure(default_name)
        self._connect()
        self._redraw()

    # ----- Properties -----

    @property
    def num_points(self) -> int:
        return len(self.L_values)

    @property
    def level(self) -> int:
        """Index of the L* level that is being chosen."""
        return len(self.points)

    @property
    def finished(self) -> bool:
        return self.level >= self.num_points

    # ----- Figure layout -----

    def _build_figure(self, default_name: str):
        self.fig = plt.figure(figsize=(10.5, 7.6))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("perfect-cmaps: create colormap")

        grid = GridSpec(
            2, 2, figure=self.fig,
            height_ratios=[1.0, 0.075], width_ratios=[1.0, 0.36],
            left=0.07, right=0.97, top=0.93, bottom=0.15, hspace=0.42, wspace=0.12,
        )
        side = grid[0, 1].subgridspec(2, 1, height_ratios=[0.32, 1.0], hspace=0.3)

        # Main slice
        self.ax_slice = self.fig.add_subplot(grid[0, 0])
        self.ax_slice.set_aspect("equal")
        self.slice_image = self.ax_slice.imshow(
            np.zeros((2, 2, 3)), origin="lower", interpolation="nearest", zorder=0
        )
        self.axis_lines = [
            self.ax_slice.axhline(0, color=ACCENT, lw=0.6, alpha=0.12, zorder=1),
            self.ax_slice.axvline(0, color=ACCENT, lw=0.6, alpha=0.12, zorder=1),
        ]
        (self.path_line,) = self.ax_slice.plot([], [], color=ACCENT, lw=1.2, alpha=0.55, zorder=2)
        self.path_points = self.ax_slice.scatter([], [], s=36, zorder=3, linewidths=1.0)
        self.last_point = self.ax_slice.scatter([], [], s=110, zorder=4, linewidths=2.0,
                                                edgecolors=ACCENT)

        # Hover readout
        self.ax_info = self.fig.add_subplot(side[0])
        self.ax_info.set_axis_off()
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.swatch = Rectangle((0.0, 0.12), 0.26, 0.76, facecolor=PANEL, edgecolor=EDGE, lw=1.0)
        self.ax_info.add_patch(self.swatch)
        self.info_text = self.ax_info.text(
            0.33, 0.5, "", va="center", ha="left", family="monospace", fontsize=9, color=TEXT
        )

        # Lightness profile / progress
        self.ax_progress = self.fig.add_subplot(side[1])
        self.ax_progress.set_title("Lightness profile", loc="left")
        self.ax_progress.set_xlim(-3, 103)
        self.ax_progress.set_ylim(self.num_points - 0.5, -0.5)
        self.ax_progress.set_xlabel("L*")
        self.ax_progress.set_yticks([])
        for spine in ("top", "right", "left"):
            self.ax_progress.spines[spine].set_visible(False)
        levels = np.arange(self.num_points)
        self.ax_progress.plot(self.L_values, levels, color=EDGE, lw=1.0, zorder=1)
        self.progress_markers = self.ax_progress.scatter(
            self.L_values, levels, s=42, zorder=2, linewidths=1.0
        )
        self.progress_current = self.ax_progress.scatter([], [], s=150, zorder=3,
                                                         facecolors="none", edgecolors=ACCENT, linewidths=1.8)

        # Colormap preview
        self.ax_preview = self.fig.add_subplot(grid[1, :])
        self.ax_preview.set_xticks([])
        self.ax_preview.set_yticks([])
        self.preview_image = self.ax_preview.imshow(
            np.zeros((1, PREVIEW_SAMPLES, 3)), aspect="auto", interpolation="bilinear",
            extent=[0, 1, 0, 1],
        )
        self.preview_title = self.ax_preview.set_title("", loc="left", fontsize=11.5)

        # Help line / status
        self.help_text = self.fig.text(0.07, 0.065, HELP_TEXT, color=MUTED, fontsize=9, va="center")

        # Save controls, shown when all levels are chosen
        self.ax_name = self.fig.add_axes([0.13, 0.04, 0.43, 0.055])
        self.default_name = default_name
        self.name_box = TextBox(self.ax_name, "Name  ", initial="",
                                color=PANEL, hovercolor="#333337", label_pad=0.02)
        self.name_box.label.set_color(TEXT)
        self.name_box.text_disp.set_color(TEXT)
        self.name_box.text_disp.set_family("monospace")
        # Greyed-out hint, shown while the field is empty; the default is used if nothing is typed
        self.name_placeholder = self.ax_name.text(
            *self.name_box.text_disp.get_position(), default_name,
            transform=self.ax_name.transAxes, va="center", ha="left",
            family="monospace", style="italic", color=PLACEHOLDER,
        )
        self.name_box.on_text_change(self._update_placeholder)
        self.ax_save = self.fig.add_axes([0.6, 0.04, 0.12, 0.055])
        self.save_button = Button(self.ax_save, "Save", color="#2f6f3e", hovercolor="#3c8a4e")
        self.ax_discard = self.fig.add_axes([0.74, 0.04, 0.12, 0.055])
        self.discard_button = Button(self.ax_discard, "Discard", color="#5a2a2a", hovercolor="#733636")
        for button in (self.save_button, self.discard_button):
            button.label.set_color(TEXT)
            button.label.set_fontweight("semibold")
        self.status_text = self.fig.text(0.07, 0.065, "", color=TEXT, fontsize=10, va="center")
        self._set_controls_visible(False)

    def _connect(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.save_button.on_clicked(lambda _: self.save())
        self.discard_button.on_clicked(lambda _: self.discard())

    def _update_placeholder(self, text: str):
        self.name_placeholder.set_visible(not text)
        self.fig.canvas.draw_idle()

    def _set_controls_visible(self, visible: bool):
        for ax in (self.ax_name, self.ax_save, self.ax_discard):
            ax.set_visible(visible)
        self.status_text.set_visible(visible)

    # ----- Gamut data -----

    def _boundary(self, level: int) -> GamutBoundary:
        if level not in self._boundaries:
            self._boundaries[level] = gamut_boundary(self.L_values[level])
        return self._boundaries[level]

    def _image(self, level: int, view: str) -> np.ndarray:
        key = (level, view)
        if key not in self._images:
            if view == "square":
                self._images[key] = render_square(self._boundary(level), self.num_bins)
            else:
                self._images[key] = render_lab(self.L_values[level], self.num_bins, LAB_EXTENT)
        return self._images[key]

    def _point_colors(self) -> np.ndarray:
        if not self.points:
            return np.zeros((0, 3))
        a, b = np.array(self.points).T
        return _lab_to_display_rgb(self.L_values[:len(self.points)], a, b)

    def _to_view(self, a: np.ndarray, b: np.ndarray, level: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map (a*, b*) to coordinates in the current view of ``level``.
        Also returns whether each point is in gamut at that level."""
        if self.finished:
            return a, b, np.ones(len(a), dtype=bool)  # Overview in the a*b* plane
        boundary = self._boundary(level)
        inside = relative_chroma(a, b, boundary) <= 1.0 + 1e-9
        if self.view == "lab":
            return a, b, inside
        u, v = ab_to_square(a, b, boundary)
        # Points that are out of gamut at this lightness are drawn on the square's edge
        scale = np.maximum(np.maximum(np.abs(u), np.abs(v)), 1.0)
        return u / scale, v / scale, inside

    def _from_view(self, x: float, y: float, level: int, exact: bool) -> Tuple[float, float]:
        """Map a position in the current view to an in-gamut (a*, b*) at ``level``."""
        L = self.L_values[level]
        if self.view == "square":
            if exact:
                return square_to_ab_exact(x, y, L)
            a, b = square_to_ab(np.clip(x, -1, 1), np.clip(y, -1, 1), self._boundary(level))
            return float(a), float(b)
        if exact:
            return clamp_to_gamut(x, y, L)
        return float(x), float(y)

    # ----- Drawing -----

    def _redraw(self):
        self._draw_slice()
        self._draw_progress()
        self._draw_preview()
        self.fig.canvas.draw_idle()

    def _draw_slice(self):
        level = min(self.level, self.num_points - 1)
        ax = self.ax_slice
        self.slice_image.set_visible(not self.finished)

        if self.finished:
            # Overview of the whole path; a single slice can't show colors of all lightness levels
            extent = [-LAB_EXTENT, LAB_EXTENT, -LAB_EXTENT, LAB_EXTENT]
            ax.set_xlabel("a*")
            ax.set_ylabel("b*")
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            ax.set_title(f"All {self.num_points} colors chosen  ·  path in the a*b* plane", loc="left")
        elif self.view == "lab":
            self.slice_image.set_data(self._image(level, self.view))
            extent = [-LAB_EXTENT, LAB_EXTENT, -LAB_EXTENT, LAB_EXTENT]
            ax.set_xlabel("a*")
            ax.set_ylabel("b*")
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            subtitle = "a*b* plane"
        else:
            self.slice_image.set_data(self._image(level, self.view))
            extent = [-1, 1, -1, 1]
            ax.set_xlabel("hue angle preserved  ·  distance from center = relative chroma")
            ax.set_ylabel("")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            subtitle = "gamut stretched to square"
        self.slice_image.set_extent(extent)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])

        if not self.finished:
            ax.set_title(
                f"Color {self.level + 1} of {self.num_points}  ·  L* = {self.L_values[level]:.1f}  ·  {subtitle}",
                loc="left",
            )

        # Path of chosen points, mapped into this slice
        if self.points:
            a, b = np.array(self.points).T
            x, y, inside = self._to_view(a, b, level)
            colors = self._point_colors()
            face = np.column_stack((colors, np.where(inside, 1.0, 0.0)))
            edge = np.where(inside[:, None], [[0.1, 0.1, 0.1, 0.9]], np.column_stack((colors, np.ones(len(colors)))))
            self.path_line.set_data(x, y)
            self.path_points.set_offsets(np.column_stack((x, y)))
            self.path_points.set_facecolors(face)
            self.path_points.set_edgecolors(edge)
            if self.finished:
                self.last_point.set_offsets(np.zeros((0, 2)))
            else:
                self.last_point.set_offsets([[x[-1], y[-1]]])
                self.last_point.set_facecolors([face[-1]])
        else:
            self.path_line.set_data([], [])
            self.path_points.set_offsets(np.zeros((0, 2)))
            self.last_point.set_offsets(np.zeros((0, 2)))

    def _draw_progress(self):
        n_done = len(self.points)
        face = np.zeros((self.num_points, 4))
        edge = np.tile(np.array([[0.56, 0.56, 0.58, 1.0]]), (self.num_points, 1))
        if n_done:
            face[:n_done, :3] = self._point_colors()
            face[:n_done, 3] = 1.0
            edge[:n_done] = [0.1, 0.1, 0.1, 0.9]
        self.progress_markers.set_facecolors(face)
        self.progress_markers.set_edgecolors(edge)
        if self.finished:
            self.progress_current.set_offsets(np.zeros((0, 2)))
        else:
            self.progress_current.set_offsets([[self.L_values[self.level], self.level]])

    def _draw_preview(self):
        if self.finished:
            self.preview_image.set_data(self.final_rgb[None])
            self.preview_title.set_text("Final colormap  (smoothed, lightness optimized)")
            return

        strip = np.tile(np.array(PANEL_RGB), (PREVIEW_SAMPLES, 1))
        if self.points:
            positions = np.linspace(0, 1, PREVIEW_SAMPLES)
            point_positions = np.linspace(0, 1, self.num_points)[:len(self.points)]
            a, b = np.array(self.points).T
            shown = positions <= point_positions[-1] + 1e-12
            if len(self.points) == 1:
                ab = np.column_stack((a, b)).repeat(shown.sum(), axis=0)
            else:
                kind = "quadratic" if len(self.points) >= 3 else "linear"
                ab = interp1d(point_positions, np.column_stack((a, b)), kind=kind, axis=0)(positions[shown])
            L = get_lightness_profile(PREVIEW_SAMPLES, self.lightness)[shown]
            strip[shown] = _lab_to_display_rgb(L, ab[:, 0], ab[:, 1])
        self.preview_image.set_data(strip[None])
        self.preview_title.set_text("Preview  (intended lightness; optimized when finished)")

    # ----- Actions -----

    def choose(self, x: float, y: float):
        """Choose the color at view position (x, y) for the current level."""
        if self.finished:
            return
        self.points.append(self._from_view(x, y, self.level, exact=True))
        if self.finished:
            self._finish()
        self._redraw()

    def undo(self):
        if not self.points:
            return
        if self.finished:
            self._set_controls_visible(False)
            self.help_text.set_visible(True)
        self.points.pop()
        self._redraw()

    def toggle_view(self):
        self.view = "square" if self.view == "lab" else "lab"
        self._redraw()

    def _finish(self):
        lab = interpolate_lab(np.array(self.points), num_values=PREVIEW_SAMPLES, profile=self.lightness)
        lab[:, 1:] = gaussian_filter1d(lab[:, 1:], sigma=PREVIEW_SAMPLES * FINAL_SMOOTHING, axis=0)
        self.final_rgb = np.clip(rgb_renormalized_lightness(lab, self.lightness), 0.0, 1.0)
        self.help_text.set_visible(False)
        self.status_text.set_text("")
        self.swatch.set_facecolor(PANEL)
        self.info_text.set_text("")
        self._set_controls_visible(True)
        # Focus the name field so the user can type a name and press Enter right away
        self.name_box.begin_typing()
        self.name_box.cursor_index = len(self.name_box.text)
        self.name_box._rendercursor()

    def save(self) -> Optional[str]:
        if not self.finished or self.saved_name is not None:
            return self.saved_name
        name = self.name_box.text.strip() or self.default_name
        points_dict = {"points": np.array(self.points).tolist(), "lightness": self.lightness}
        # Precompute the default lightness fit so that get_cmap doesn't need to optimize
        ensure_fit(points_dict, self.lightness, DEFAULT_INTERPOLATION, None)
        self.saved_name = save_data(points_dict, name)
        self.status_text.set_text(f"Saved as '{self.saved_name}'. Use get_cmap('{self.saved_name}').")
        for ax in (self.ax_name, self.ax_save, self.ax_discard):
            ax.set_visible(False)
        self.fig.canvas.draw_idle()

        timer = self.fig.canvas.new_timer(interval=1500)
        timer.add_callback(plt.close, self.fig)
        timer.single_shot = True
        timer.start()
        self._close_timer = timer  # Keep a reference so it isn't garbage collected
        return self.saved_name

    def discard(self):
        plt.close(self.fig)

    # ----- Events -----

    def _navigating(self) -> bool:
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        return toolbar is not None and bool(getattr(toolbar, "mode", ""))

    def _on_click(self, event):
        if event.inaxes is not self.ax_slice or self._navigating():
            return
        if event.button == 1 and event.xdata is not None:
            self.choose(event.xdata, event.ydata)
        elif event.button == 3:
            self.undo()

    def _on_key(self, event):
        if self.name_box.capturekeystrokes:  # Typing a name
            if event.key in ("enter", "return"):
                self.name_box.stop_typing()
                self.save()
            elif event.key == "escape":
                self.name_box.stop_typing()
            return
        if event.key == "backspace":
            self.undo()
        elif event.key == "v":
            self.toggle_view()
        elif event.key == "escape":
            plt.close(self.fig)

    def _on_move(self, event):
        if event.inaxes is not self.ax_slice or event.xdata is None or self.finished:
            return
        level = min(self.level, self.num_points - 1)
        L = self.L_values[level]
        x, y = event.xdata, event.ydata
        if self.view == "square" and max(abs(x), abs(y)) > 1:
            return
        a, b = self._from_view(x, y, level, exact=False)
        rho = float(relative_chroma(a, b, self._boundary(level)))
        rgb = Lab_to_sRGB(np.array([[L, a, b]]))[0]
        in_gamut = bool(np.all((rgb >= -1e-6) & (rgb <= 1 + 1e-6)))
        rgb = np.clip(rgb, 0.0, 1.0)

        self.swatch.set_facecolor(rgb if in_gamut else PANEL)
        lines = [
            f"L*  {L:6.1f}",
            f"a*  {a:6.1f}",
            f"b*  {b:6.1f}",
            f"C   {100 * rho:5.0f} %" if rho <= 1.0 else "C       –",
            _hex(rgb) if in_gamut else "out of gamut",
        ]
        self.info_text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()


def parse_args():
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument(
        "--num_points",
        "-n",
        type=int,
        default=20,
        help="Number of control points to interpolate between."
    )
    arg_parser.add_argument(
        "--lightness",
        "-l",
        type=str,
        default="linear",
        help="The lightness profile of the colormap. Allowed values are: " + ", ".join(SUPPORTED_L_PROFILES) + "."
    )
    arg_parser.add_argument(
        "--view",
        "-v",
        type=str,
        default="lab",
        choices=VIEWS,
        help="Initial view of the gamut slices: 'lab' shows the a*b* plane, 'square' stretches "
             "each slice to fill the square. Press V in the window to switch."
    )
    arg_parser.add_argument(
        "--help",
        "-h",
        action="help",
        default=argparse.SUPPRESS,
        help="Interactively create a perceptually uniform colormap by choosing one color per lightness level."
    )
    return arg_parser.parse_args()


def create_cmap(
        num_points: int = 20,
        lightness: str = "linear",
        view: str = "lab",
        num_bins: int = 500
    ) -> Optional[str]:
    """Lets the user define control points for a La*b* colormap
    by iterating through a*b* slices for fixed L values.
    The colormap is interpolated in La*b* space between these control points.

    In the window: click to choose a color, right-click or Backspace to undo,
    V to switch view, Esc to quit. When all colors are chosen, the final colormap
    is shown and can be saved under a name.

    Args:
        num_points (int, optional): Number of control points. Defaults to 20.
        lightness (str, optional): Lightness profile for the colormap. Defaults to "linear".
        view (str, optional): Initial view: "lab" shows the a*b* plane, "square" stretches
            each gamut slice radially to fill the square, preserving hue, so that the distance
            from the center is the chroma relative to the most saturated color of that hue.
            Defaults to "lab".
        num_bins (int, optional): Resolution of the slice images. Defaults to 500.

    Returns:
        Optional[str]: The name the colormap was saved as, or None if it wasn't saved.
    """
    with plt.rc_context(STYLE):
        creator = ColormapCreator(num_points, lightness, view, num_bins)
        plt.show()
    return creator.saved_name


if __name__ == "__main__":
    args = parse_args()
    create_cmap(args.num_points, args.lightness, args.view)
