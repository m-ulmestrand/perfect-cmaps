"""Conversions between sRGB, CIE XYZ and CIE L*a*b*, with the D65 white point.

Value ranges: sRGB and XYZ in [0, 1] (Y = 1 for the white point), L* in [0, 100].
Out-of-gamut colors are not clipped, so callers can detect them.
"""
import numpy as np


def _xy_to_XYZ(xy) -> np.ndarray:
    x, y = xy
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


# CIE 1931 2° D65 white point, used by both sRGB and L*a*b* here
D65_XY = (0.3127, 0.3290)
D65_XYZ = _xy_to_XYZ(D65_XY)

# sRGB primaries (IEC 61966-2-1)
SRGB_PRIMARIES_XY = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))


def _normalised_primary_matrix(primaries_xy, white_XYZ: np.ndarray) -> np.ndarray:
    """RGB -> XYZ matrix for the given primaries, such that RGB = (1, 1, 1) maps to the white point."""
    primaries = np.column_stack([_xy_to_XYZ(p) for p in primaries_xy])
    scale = np.linalg.solve(primaries, white_XYZ)
    return primaries * scale


SRGB_TO_XYZ_MATRIX = _normalised_primary_matrix(SRGB_PRIMARIES_XY, D65_XYZ)
XYZ_TO_SRGB_MATRIX = np.linalg.inv(SRGB_TO_XYZ_MATRIX)

# CIE constants for L*a*b*
_EPSILON = 216 / 24389
_KAPPA = 24389 / 27


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Linear RGB -> gamma-encoded sRGB."""
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0031308), 1 / 2.4) - 0.055,
    )


def _srgb_decode(encoded: np.ndarray) -> np.ndarray:
    """Gamma-encoded sRGB -> linear RGB."""
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((np.maximum(encoded, 0.04045) + 0.055) / 1.055, 2.4),
    )


def sRGB_to_XYZ(rgb: np.ndarray) -> np.ndarray:
    return _srgb_decode(np.asarray(rgb, dtype=float)) @ SRGB_TO_XYZ_MATRIX.T


def XYZ_to_sRGB(XYZ: np.ndarray) -> np.ndarray:
    return _srgb_encode(np.asarray(XYZ, dtype=float) @ XYZ_TO_SRGB_MATRIX.T)


def XYZ_to_Lab(XYZ: np.ndarray) -> np.ndarray:
    t = np.asarray(XYZ, dtype=float) / D65_XYZ
    f = np.where(t > _EPSILON, np.cbrt(t), (_KAPPA * t + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack((L, a, b), axis=-1)


def Lab_to_XYZ(Lab: np.ndarray) -> np.ndarray:
    Lab = np.asarray(Lab, dtype=float)
    L, a, b = Lab[..., 0], Lab[..., 1], Lab[..., 2]
    f_y = (L + 16) / 116
    f_x = f_y + a / 500
    f_z = f_y - b / 200

    def f_inverse(f):
        return np.where(f ** 3 > _EPSILON, f ** 3, (116 * f - 16) / _KAPPA)

    x = f_inverse(f_x)
    y = np.where(L > _KAPPA * _EPSILON, f_y ** 3, L / _KAPPA)
    z = f_inverse(f_z)
    return np.stack((x, y, z), axis=-1) * D65_XYZ


def Lab_to_sRGB(Lab: np.ndarray) -> np.ndarray:
    """CIE L*a*b* -> sRGB, without clipping out-of-gamut values."""
    return XYZ_to_sRGB(Lab_to_XYZ(Lab))


def sRGB_to_Lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB -> CIE L*a*b*."""
    return XYZ_to_Lab(sRGB_to_XYZ(rgb))
