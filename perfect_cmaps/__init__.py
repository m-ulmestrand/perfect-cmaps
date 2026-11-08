from .colormaps import get_cmap, compare_cmaps
from .color_utils import plot_cmap
from .storage import list_cmaps


__all__ = ["get_cmap", "create_cmap", "list_cmaps", "compare_cmaps", "plot_cmap"]
__version__ = "2.0.0"


def __getattr__(name: str):
    # create_cmap pulls in matplotlib.pyplot and its widgets, so it is only imported when used
    if name == "create_cmap":
        from .customize_cmap import create_cmap
        return create_cmap
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
