"""Shared matplotlib styling for documentation figures.

Figures appear on both the light and the dark theme of the same page, rendered
once as a single PNG. So the style is built to read on *either* background:

* transparent figure/axes — the page background (white or navy) shows through,
  instead of a glaring white panel floating in dark mode;
* steel-grey text, spines, ticks, and grid — mid-tone, legible on white and on
  deep navy alike;
* data drawn in the brand's amber "transfer burn" and steel, both of which keep
  their contrast against either background.

Call :func:`apply` at the top of a plot script, then pull colors from the module
constants (``AMBER``, ``STEEL`` …) for individual artists.
"""

import matplotlib as mpl
from cycler import cycler

# -- Brand figure palette (legible on white *and* navy) --------------------
AMBER = "#e6a23c"  # primary data — the "burn"
AMBER_BRIGHT = "#f0b441"  # emphasis / target marker
STEEL = "#7d89ab"  # secondary data — guess, second series
STEEL_DARK = "#5c6a8f"  # markers that need a touch more weight
FAINT = "#9aa3bd"  # reference geometry, very low alpha
INK = "#6b7798"  # text, spines, ticks — mid steel, theme-agnostic
GRID = "#8a93b3"  # grid lines (used at low alpha)


def apply():
    """Install the Tycho figure style into matplotlib's global rcParams."""
    mpl.rcParams.update(
        {
            # Let the page background show through (no white box in dark mode).
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            # Mid-steel chrome reads on both themes.
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.8,
            "grid.color": GRID,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.labelcolor": INK,
            "font.size": 10,
            # Default series order: burn first, steel second.
            "axes.prop_cycle": cycler(color=[AMBER, STEEL, AMBER_BRIGHT, STEEL_DARK]),
        }
    )
