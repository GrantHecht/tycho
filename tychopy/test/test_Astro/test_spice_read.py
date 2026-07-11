import pytest

import tychopy.astro.spice_read as spice_read


def test_sample_jds_includes_endpoint():
    """Characterizes the fencepost fix.

    Pre-fix, the inline expression used in GetEphemTraj2/PoleVector/
    GetEphemTraj was ``x * (endJD - startJD) / numstep + startJD`` for
    ``x in range(numstep)``. For (startJD=2451545.0, endJD=2451555.0,
    numstep=11) the last sample (x=10) evaluated to
    ``10 * 10.0 / 11 + 2451545.0 == 2451554.0909...`` -- one full step
    (~0.909 day) short of endJD, rather than the intended inclusive
    endpoint. ``_sample_jds`` uses denominator ``numstep - 1`` so the
    last sample lands exactly on ``endJD``.
    """
    jds = spice_read._sample_jds(2451545.0, 2451555.0, 11)
    assert jds[0] == 2451545.0
    assert jds[-1] == 2451555.0
    assert len(jds) == 11


def test_sample_jds_numstep_less_than_two_raises():
    with pytest.raises(ValueError):
        spice_read._sample_jds(2451545.0, 2451555.0, 1)


def test_sample_jds_uniform_spacing():
    startJD = 2451545.0
    endJD = 2451555.0
    numstep = 11
    jds = spice_read._sample_jds(startJD, endJD, numstep)

    assert len(jds) == numstep
    assert jds[0] == startJD
    assert jds[-1] == endJD

    step = (endJD - startJD) / float(numstep - 1)
    for i, jd in enumerate(jds):
        assert jd == pytest.approx(startJD + i * step, rel=1e-12)
