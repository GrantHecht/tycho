import datetime as pydt

import numpy as np
import pytest

import tychopy.astro.date as date


def test_timedelta_to_days_microseconds():
    td = pydt.timedelta(microseconds=500000)  # 0.5 s
    assert date.timedelta_to_days(td) == pytest.approx(0.5 / 86400.0, rel=1e-12)


def test_datetime_plus_fractional_second_timedelta():
    d = date.datetime(2020, 5, 17, 12, 0, 0)
    d2 = d + pydt.timedelta(seconds=1.5)
    ref = pydt.datetime(2020, 5, 17, 12, 0, 1, 500000)
    # datetime.__add__ round-trips through Julian Day (a ~2.459e6-magnitude
    # double whose ULP is ~5.5e-10 days, i.e. ~47 microseconds, at this
    # date) by the class's documented design, so exact microsecond equality
    # isn't achievable here -- this is JD double-precision noise, not the
    # timedelta_to_days scale bug. Allow a tolerance sized to that ULP.
    assert (d2.hour, d2.minute, d2.second) == (ref.hour, ref.minute, ref.second)
    assert abs(d2.microsecond - ref.microsecond) <= 100


def test_days_to_hmsm_never_returns_full_second_micro():
    h, m, s, us = date.days_to_hmsm(0.999999999999)
    assert 0 <= us <= 999999


def test_jd_to_datetime_day_boundary_roundtrip(monkeypatch):
    """Exercise jd_to_datetime's hour==24 day-rollover carry.

    Real doubles can't actually land in the carry window here: at
    epoch-scale JD magnitude (~2.4589e6) the ULP is ~40us, far coarser than
    the ~0.5us of fractional-day slack needed to trip days_to_hmsm's
    round-up-to-24h path, so `datetime_to_jd(...) + 0.999999999999` collapses
    to exactly +1.0 rather than ever reaching the carry. So instead we
    monkeypatch days_to_hmsm -- a module-global lookup from inside
    jd_to_datetime, not a from-import binding, so patching the module
    attribute is sufficient -- to force it to return hour=24 directly, then
    verify jd_to_datetime normalizes that into the next calendar day. A
    trailing call on the real, unpatched path is kept only as a no-crash
    sanity check; it does not exercise the carry.
    """
    monkeypatch.setattr(date, "days_to_hmsm", lambda frac_days: (24, 0, 0, 0))
    jd = date.datetime_to_jd(date.datetime(2020, 1, 1))
    d = date.jd_to_datetime(jd)
    assert isinstance(d, date.datetime)
    assert (d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond) == (
        2020,
        1,
        2,
        0,
        0,
        0,
        0,
    )

    monkeypatch.undo()
    jd_unpatched = date.datetime_to_jd(date.datetime(2020, 1, 1)) + 0.999999999999
    d_unpatched = date.jd_to_datetime(jd_unpatched)  # must not raise
    assert isinstance(d_unpatched, date.datetime)


def test_datetime_jd_roundtrip_sweep():
    for jd0 in np.linspace(2451545.0, 2451546.0, 97):
        d = date.jd_to_datetime(jd0)
        assert date.datetime_to_jd(d) == pytest.approx(jd0, abs=1e-8)


def test_seven_arg_datetime_to_jd_removed():
    import inspect

    assert len(inspect.signature(date.datetime_to_jd).parameters) == 1
