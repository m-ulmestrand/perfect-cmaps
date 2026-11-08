"""Geometry of sRGB gamut slices (fixed L*) in the a*b* plane, used by the colormap creator.

Every slice is star-shaped around the neutral axis (a* = b* = 0): along each hue ray from
the origin, colors stay in gamut up to the boundary radius R(theta) and are out of gamut
beyond it. That makes it possible to stretch a slice radially so that it exactly fills
the square [-1, 1]^2:

    a point (u, v) in the square has hue angle theta = atan2(v, u) and
    relative chroma rho = max(|u|, |v|) in [0, 1], and maps to
    (a*, b*) = rho * R(theta) * (cos(theta), sin(theta)).

The hue angle is preserved, and rho = 1 is the most saturated in-gamut color for that
hue and lightness. Because this is computed as an inverse mapping (square -> a*b*),
every pixel gets an exact color, without holes.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from perfect_cmaps.color_utils import Lab_to_sRGB, in_gamut


# Chroma beyond which no sRGB color exists at any lightness (max is about 134)
MAX_CHROMA = 150.0
# Half-width of the plain a*b* view. The sRGB gamut spans a* in [-86.2, 98.2] and
# b* in [-107.9, 94.5] over all lightness levels, so this shows all of it at every level.
LAB_EXTENT = 110.0


@dataclass(frozen=True)
class GamutBoundary:
    """Boundary radius of a gamut slice as a function of hue angle."""
    L: float
    theta: np.ndarray   # Increasing angles in [-pi, pi)
    radius: np.ndarray  # Boundary chroma at each angle

    def radius_at(self, theta: np.ndarray) -> np.ndarray:
        """Boundary chroma at arbitrary angles, by periodic linear interpolation."""
        return np.interp(theta, self.theta, self.radius, period=2 * np.pi)


def boundary_radius(
    L: float,
    theta: np.ndarray,
    n_coarse: int = 300,
    n_bisection: int = 30,
) -> np.ndarray:
    """Exact boundary chroma of the slice at lightness ``L`` along the hue rays ``theta``.

    Each ray is first sampled coarsely to find where it first leaves the gamut, which is
    robust even for the rare rays that graze a sharp corner of the boundary. The edge is
    then refined by bisection, for all rays at once.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n_angles = theta.size
    cos, sin = np.cos(theta), np.sin(theta)

    radii = np.linspace(0.0, MAX_CHROMA, n_coarse)
    chroma = np.repeat(radii, n_angles)
    inside = in_gamut(
        np.full(chroma.size, L), chroma * np.tile(cos, n_coarse), chroma * np.tile(sin, n_coarse)
    ).reshape(n_coarse, n_angles)

    # First out-of-gamut sample along each ray; the neutral axis itself is always in gamut
    first_outside = np.argmin(inside, axis=0)
    first_outside[inside.all(axis=0)] = n_coarse - 1
    r_inside = radii[np.maximum(first_outside - 1, 0)]
    r_outside = radii[first_outside]

    L_column = np.full(n_angles, L)
    for _ in range(n_bisection):
        r_mid = (r_inside + r_outside) / 2
        mid_inside = in_gamut(L_column, r_mid * cos, r_mid * sin)
        r_inside = np.where(mid_inside, r_mid, r_inside)
        r_outside = np.where(mid_inside, r_outside, r_mid)

    return r_inside


def gamut_boundary(L: float, n_angles: int = 720) -> GamutBoundary:
    """Boundary of the slice at lightness ``L``, sampled at ``n_angles`` hue angles.
    Between the samples it is linearly interpolated, which is accurate enough for drawing;
    use ``square_to_ab_exact`` or ``clamp_to_gamut`` where a color must be in gamut."""
    theta = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    return GamutBoundary(float(L), theta, boundary_radius(L, theta))


def square_to_ab(u: np.ndarray, v: np.ndarray, boundary: GamutBoundary) -> Tuple[np.ndarray, np.ndarray]:
    """Map square coordinates (u, v) in [-1, 1]^2 to (a*, b*) in the gamut slice."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    theta = np.arctan2(v, u)
    rho = np.maximum(np.abs(u), np.abs(v))
    chroma = rho * boundary.radius_at(theta)
    return chroma * np.cos(theta), chroma * np.sin(theta)


def square_to_ab_exact(u: float, v: float, L: float) -> Tuple[float, float]:
    """Like ``square_to_ab``, but with the boundary computed exactly along this ray,
    so that the result is guaranteed to be in gamut. Meant for single points, e.g. clicks."""
    theta = np.arctan2(v, u)
    rho = min(max(abs(u), abs(v)), 1.0)
    chroma = rho * float(boundary_radius(L, theta)[0])
    return chroma * np.cos(theta), chroma * np.sin(theta)


def ab_to_square(a: np.ndarray, b: np.ndarray, boundary: GamutBoundary) -> Tuple[np.ndarray, np.ndarray]:
    """Map (a*, b*) to square coordinates, the inverse of ``square_to_ab``.
    Colors outside the gamut slice end up outside [-1, 1]^2."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    theta = np.arctan2(b, a)
    rho = np.hypot(a, b) / boundary.radius_at(theta)
    cos, sin = np.cos(theta), np.sin(theta)
    # Point on the square's edge in direction theta is (cos, sin) / max(|cos|, |sin|)
    edge = np.maximum(np.abs(cos), np.abs(sin))
    return rho * cos / edge, rho * sin / edge


def relative_chroma(a: np.ndarray, b: np.ndarray, boundary: GamutBoundary) -> np.ndarray:
    """Chroma relative to the most saturated in-gamut color with the same hue (1 = boundary)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.hypot(a, b) / boundary.radius_at(np.arctan2(b, a))


def clamp_to_gamut(a: float, b: float, L: float) -> Tuple[float, float]:
    """Pull an out-of-gamut (a*, b*) radially onto the exact boundary, keeping its hue."""
    radius = float(boundary_radius(L, np.arctan2(b, a))[0])
    chroma = float(np.hypot(a, b))
    if chroma <= radius:
        return float(a), float(b)
    return float(a) * radius / chroma, float(b) * radius / chroma


def render_square(boundary: GamutBoundary, size: int = 500) -> np.ndarray:
    """RGB image (size, size, 3) of the slice stretched to fill the square.
    Rows go from v = -1 to v = 1, so display it with ``origin="lower"``."""
    coords = (np.arange(size) + 0.5) / size * 2 - 1  # Pixel centers
    U, V = np.meshgrid(coords, coords, indexing="xy")
    a, b = square_to_ab(U.ravel(), V.ravel(), boundary)
    lab = np.column_stack((np.full(a.size, boundary.L), a, b))
    return np.clip(Lab_to_sRGB(lab), 0.0, 1.0).reshape(size, size, 3)


def render_lab(L: float, size: int = 500, extent: float = LAB_EXTENT) -> np.ndarray:
    """RGBA image (size, size, 4) of the plain a*b* slice over [-extent, extent]^2,
    with out-of-gamut pixels transparent. Display it with ``origin="lower"``."""
    coords = (np.arange(size) + 0.5) / size * 2 * extent - extent
    A, B = np.meshgrid(coords, coords, indexing="xy")
    lab = np.column_stack((np.full(A.size, L), A.ravel(), B.ravel()))
    rgb = Lab_to_sRGB(lab)
    inside = np.all((rgb >= 0.0) & (rgb <= 1.0), axis=1)
    rgba = np.zeros((A.size, 4))
    rgba[:, :3] = np.clip(rgb, 0.0, 1.0)
    rgba[:, 3] = inside
    return rgba.reshape(size, size, 4)
