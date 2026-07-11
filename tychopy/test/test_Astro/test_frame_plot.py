"""Smoke tests for tychopy.astro.frame_plot (CODEBASE 1.2).

Four live defects fixed here (all reproduce headless on matplotlib 3.10.8):

1. ``Plot3dAx`` iterated ``self.Points.items()`` (2-tuples of
   ``(name, dict)``) and then indexed the tuple with ``obj["Pos"]`` ->
   ``TypeError: tuple indices must be integers or slices, not str``.
   ``Plot2dAx`` already did this correctly via ``.values()``.
2. ``addPropTraj`` forwarded to ``self.addTraj(name, traj, ...)`` but
   ``addTraj``'s signature is ``(traj, name, ...)`` -- the arguments were
   swapped, corrupting the stored trajectory (and raising downstream once
   plotted, since a bare name string was treated as the point array).
3. ``Plot3d`` called ``fig.gca(projection="3d")``, which was removed in
   matplotlib>=3.6 (``AttributeError: 'Figure' object has no attribute
   'gca'`` semantics changed -- ``gca`` no longer accepts kwargs to create
   a new Axes) -- replaced with ``fig.add_subplot(projection="3d")``.
4. ``Arrow3D`` overrode the pre-3.5 ``draw(self, renderer)`` protocol and
   reached into ``renderer.M``, which no longer exists on modern
   ``RendererAgg`` -- replaced with the ``do_3d_projection`` protocol
   which reads the projection matrix off ``self.axes.M`` instead.

A single smoke test exercises all four: build a ``PlotBase``, add a
trajectory via both ``addTraj`` and ``addPropTraj``, add a point, plot it
on both a 2-D and a 3-D Axes, and force a canvas draw (which invokes
``Arrow3D.do_3d_projection`` for any arrow artists added by ``PlotSTraj``).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tychopy.astro.frame_plot import Arrow3D, PlotBase


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _make_plot_base():
    pb = PlotBase()
    pb.addTraj([[0, 0, 0], [1, 1, 1]], "t1", color="blue")
    pb.addPropTraj([[0, 0, 0], [1, 1, 1]], "t2", color="red")
    pb.addPoint([0.5, 0.5, 0.5], "p1", color="black")
    return pb


def test_addproptraj_forwards_args_in_traj_name_order():
    # Regression for bug 2: addPropTraj used to swap (traj, name), which
    # stored `name` (a string) as the trajectory array.
    pb = _make_plot_base()
    assert pb.Trajs["t2"]["Name"] == "t2"
    np.testing.assert_array_equal(
        pb.Trajs["t2"]["Traj"], np.array([[0, 0, 0], [1, 1, 1]]).T
    )


def test_plot2d_and_plot3d_ax_smoke():
    pb = _make_plot_base()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    pb.POI = {}
    returned2 = pb.Plot2dAx(ax2)
    assert returned2 is ax2
    fig2.canvas.draw()

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection="3d")
    # Regression for bug 1: Plot3dAx used to iterate self.Points.items()
    # (tuples) instead of .values() (dicts), raising TypeError here.
    returned3 = pb.Plot3dAx(ax3)
    assert returned3 is ax3
    # Regression for bug 4: forces a canvas draw, which calls
    # do_3d_projection() on any Arrow3D artists in the Axes.
    fig3.canvas.draw()


def test_plot3d_wrapper_smoke(monkeypatch):
    # Regression for bug 3: Plot3d used to call the removed
    # fig.gca(projection="3d") API.
    pb = _make_plot_base()
    pb.POI = {}
    monkeypatch.setattr(plt, "show", lambda: None)
    pb.Plot3d()


def test_arrow3d_do_3d_projection_executes():
    # Directly confirm the do_3d_projection protocol runs and updates the
    # arrow's 2-D screen-space positions from the 3-D vertex data, proving
    # the fixed method (not the removed draw(renderer) override) executes.
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    arrow = Arrow3D(
        [0, 1],
        [0, 1],
        [0, 1],
        mutation_scale=8,
        lw=1.2,
        arrowstyle="-|>",
        color="black",
    )
    before = list(arrow._posA_posB)
    assert before == [(0, 0), (0, 0)]

    ax.add_artist(arrow)
    fig.canvas.draw()
    after = list(arrow._posA_posB)

    # do_3d_projection() calls set_positions() with the projected 2-D
    # screen coordinates derived from ax.M -- if it never ran (e.g. the
    # removed draw(renderer) override silently no-ops under Agg), the
    # patch would retain its (0, 0)/(0, 0) placeholder positions.
    assert after != before
