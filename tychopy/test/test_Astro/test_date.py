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


def test_jd_to_datetime_day_boundary_roundtrip():
    # A JD whose fractional day rounds to exactly 1 s at micro precision
    jd = date.datetime_to_jd(date.datetime(2020, 1, 1)) + 0.999999999999
    d = date.jd_to_datetime(jd)  # must not raise
    assert isinstance(d, date.datetime)


def test_datetime_jd_roundtrip_sweep():
    for jd0 in np.linspace(2451545.0, 2451546.0, 97):
        d = date.jd_to_datetime(jd0)
        assert date.datetime_to_jd(d) == pytest.approx(jd0, abs=1e-8)


def test_seven_arg_datetime_to_jd_removed():
    import inspect

    assert len(inspect.signature(date.datetime_to_jd).parameters) == 1
