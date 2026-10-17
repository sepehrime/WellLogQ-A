from pathlib import Path

import pytest

from welllog.data import load_las

SAMPLE = Path(__file__).parent.parent / "data" / "sample.las"

MINI_LAS = """~Version Information
VERS.   2.0 :
WRAP.   NO  :
~Well Information
STRT.M  100.0 :
STOP.M  101.0 :
STEP.M  0.5 :
NULL.   -999.25 :
WELL.   Test Well :
~Curve Information
DEPT.M     : Depth
GR.GAPI    : Gamma Ray
NPHI.V/V   : Neutron Porosity
~A
100.0   50.0   0.20
100.5 -999.25  0.25
101.0   80.0 -999.25
"""


@pytest.fixture
def mini_las(tmp_path):
    path = tmp_path / "mini.las"
    path.write_text(MINI_LAS)
    return path


def test_mini_file(mini_las):
    df, curves = load_las(mini_las)
    assert list(df.index) == [100.0, 100.5, 101.0]
    assert df.attrs["depth_unit"] == "m"
    assert df.attrs["well"] == "Test Well"
    assert curves == [
        {"mnemonic": "GR", "unit": "GAPI", "description": "Gamma Ray"},
        {"mnemonic": "NPHI", "unit": "V/V", "description": "Neutron Porosity"},
    ]
    assert df.isna().sum().sum() == 2
    assert (df == -999.25).sum().sum() == 0


def test_sample_file():
    df, curves = load_las(SAMPLE)
    assert df.attrs["depth_unit"] == "ft"
    assert df.index[0] == 345.0 and df.index[-1] == 5209.0
    assert [c["mnemonic"] for c in curves] == list(df.columns)
    assert (df == -999.25).sum().sum() == 0
