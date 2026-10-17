from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from welllog.data import load_las
from welllog.tools import WellLogError, curve_stats, flag_zones, list_curves

SAMPLE = Path(__file__).parent.parent / "data" / "sample.las"


@pytest.fixture
def df():
    """100–110 m at 0.5 m: constant NPHI of 0.20, GR 50 with one spike of 150 at 104–105 m."""
    depth = np.arange(100.0, 110.5, 0.5)
    gr = np.full(depth.size, 50.0)
    gr[(depth >= 104.0) & (depth <= 105.0)] = 150.0
    frame = pd.DataFrame({"GR": gr, "NPHI": 0.20}, index=pd.Index(depth, name="DEPT"))
    frame.attrs.update(depth_unit="m", well="Synthetic", units={"GR": "GAPI", "NPHI": "V/V"})
    return frame


@pytest.fixture
def curves():
    return [
        {"mnemonic": "GR", "unit": "GAPI", "description": "Gamma Ray"},
        {"mnemonic": "NPHI", "unit": "V/V", "description": "Neutron Porosity"},
    ]


def test_list_curves(df, curves):
    info = list_curves(df, curves)
    assert (info["top"], info["base"], info["step"], info["depth_unit"]) == (100.0, 110.0, 0.5, "m")
    assert [c["mnemonic"] for c in info["curves"]] == ["GR", "NPHI"]
    assert info["curves"][0]["unit"] == "GAPI"
    assert info["curves"][0]["valid_samples"] == 21


def test_constant_porosity_mean_is_exact(df):
    stats = curve_stats(df, "NPHI")
    assert stats["mean"] == 0.20
    assert stats["min"] == stats["max"] == stats["p10"] == stats["p50"] == stats["p90"] == 0.20
    assert stats["count"] == 21
    assert stats["unit"] == "V/V"


def test_curve_stats_interval_and_case_insensitive(df):
    stats = curve_stats(df, "gr", top=103.0, base=106.0)  # 7 samples, 3 of them 150
    assert stats["curve"] == "GR"
    assert (stats["top"], stats["base"], stats["count"]) == (103.0, 106.0, 7)
    assert stats["mean"] == pytest.approx((4 * 50 + 3 * 150) / 7)
    assert (stats["min"], stats["max"]) == (50.0, 150.0)


def test_curve_stats_ignores_nan(df):
    df.loc[100.0:101.0, "NPHI"] = np.nan
    assert curve_stats(df, "NPHI")["count"] == 18


def test_curve_stats_trims_partial_interval(df):
    stats = curve_stats(df, "GR", top=90.0, base=102.0)
    assert (stats["top"], stats["base"]) == (100.0, 102.0)


def test_one_gr_spike_gives_one_zone(df):
    zones = flag_zones(df, "GR", threshold=100)
    assert zones == [{"top": 104.0, "base": 105.0, "thickness": 1.5, "mean": 150.0}]


def test_flag_zones_below_and_min_thickness(df):
    below = flag_zones(df, "GR", threshold=100, above=False)
    assert [(z["top"], z["base"]) for z in below] == [(100.0, 103.5), (105.5, 110.0)]
    assert flag_zones(df, "GR", threshold=100, min_thickness=2.0) == []


def test_flag_zones_nan_splits_zone(df):
    df.loc[104.5, "GR"] = np.nan
    zones = flag_zones(df, "GR", threshold=100)
    assert [(z["top"], z["base"]) for z in zones] == [(104.0, 104.0), (105.0, 105.0)]


@pytest.mark.parametrize("call, message", [
    (lambda d: curve_stats(d, "RHOB"), "Unknown curve 'RHOB'. Available curves: GR, NPHI"),
    (lambda d: curve_stats(d, "GR", top=200, base=300), "outside the logged depth range 100–110 m"),
    (lambda d: curve_stats(d, "GR", top=108, base=102), "Top (108) must be shallower than base (102)"),
    (lambda d: flag_zones(d, "XX", threshold=1), "Unknown curve 'XX'"),
    (lambda d: flag_zones(d, "GR", threshold=1, top=0, base=50), "outside the logged depth range"),
])
def test_bad_input_gives_clear_error(df, call, message):
    with pytest.raises(WellLogError, match=message.replace("(", r"\(").replace(")", r"\)")):
        call(df)


def test_curve_stats_no_valid_samples(df):
    df["NPHI"] = np.nan
    with pytest.raises(WellLogError, match="no valid samples"):
        curve_stats(df, "NPHI")


def test_curve_stats_matches_pandas_on_sample_file():
    real, _ = load_las(SAMPLE)
    stats = curve_stats(real, "CNLS", top=2500, base=2700)
    assert stats["mean"] == pytest.approx(real.loc[2500:2700, "CNLS"].mean(), rel=1e-12)
    assert stats["count"] == real.loc[2500:2700, "CNLS"].count()
    assert stats["unit"] == "PU" and stats["depth_unit"] == "ft"
