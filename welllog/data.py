"""Load a LAS file into a depth-indexed DataFrame plus curve metadata."""

from pathlib import Path

import lasio
import numpy as np
import pandas as pd

DEFAULT_NULL = -999.25

_DEPTH_UNITS = {
    "M": "m", "METER": "m", "METERS": "m", "METRE": "m", "METRES": "m",
    "F": "ft", "FT": "ft", "FEET": "ft", "FOOT": "ft",
}


def _depth_unit(las: lasio.LASFile) -> str | None:
    """Return "m" or "ft" from the depth curve or STRT unit, or None if unknown."""
    for raw in (las.curves[0].unit, las.well["STRT"].unit if "STRT" in las.well else ""):
        unit = _DEPTH_UNITS.get((raw or "").strip().upper())
        if unit:
            return unit
    return None


def load_las(path: str | Path) -> tuple[pd.DataFrame, list[dict]]:
    """Read a LAS file.

    Returns ``(df, curves)``:
      - ``df`` is indexed by depth, one column per curve, with null values as NaN.
        ``df.attrs["depth_unit"]`` is "m", "ft" or None; ``df.attrs["well"]`` is the WELL name.
      - ``curves`` is a list of ``{mnemonic, unit, description}`` for every non-depth curve,
        in file order.
    """
    las = lasio.read(str(path))
    df = las.df()

    nulls = {DEFAULT_NULL}
    if "NULL" in las.well:
        try:
            nulls.add(float(las.well["NULL"].value))
        except (TypeError, ValueError):
            pass
    df = df.replace(list(nulls), np.nan)

    depth_unit = _depth_unit(las)
    df.index.name = las.curves[0].mnemonic
    df.attrs["depth_unit"] = depth_unit
    df.attrs["well"] = str(las.well["WELL"].value) if "WELL" in las.well else None

    curves = [
        {"mnemonic": c.mnemonic, "unit": c.unit.strip(), "description": c.descr.strip()}
        for c in las.curves[1:]
    ]
    return df, curves
