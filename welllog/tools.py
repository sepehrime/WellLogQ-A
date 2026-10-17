"""Analysis tools over a depth-indexed well-log DataFrame (as returned by ``load_las``).

Every function returns plain Python types (JSON-safe). Bad input raises ``WellLogError``
with a message meant to be shown to the user or the LLM.

Conventions:
  - ``top``/``base`` are inclusive depths in the file's depth unit; top must be shallower.
    An interval that only partly overlaps the logged range is trimmed to it, and the
    interval actually used is reported back.
  - P10/P50/P90 are the 10th/50th/90th statistical percentiles (P10 is the low value).
"""

import math

import numpy as np
import pandas as pd


class WellLogError(ValueError):
    """Invalid request against the well data (unknown curve, bad interval, no data)."""


def _f(x) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df if df.index.is_monotonic_increasing else df.sort_index()


def _step(df: pd.DataFrame) -> float:
    diffs = np.diff(df.index.to_numpy(dtype=float))
    return float(np.median(diffs)) if len(diffs) else 0.0


def _unit(df: pd.DataFrame, curve: str) -> str:
    return df.attrs.get("units", {}).get(curve, "")


def resolve_curve(df: pd.DataFrame, curve: str) -> str:
    """Return the column name matching ``curve`` (case-insensitive), or raise."""
    if curve in df.columns:
        return curve
    matches = [c for c in df.columns if c.upper() == str(curve).strip().upper()]
    if matches:
        return matches[0]
    raise WellLogError(
        f"Unknown curve '{curve}'. Available curves: {', '.join(map(str, df.columns))}."
    )


def _interval(df: pd.DataFrame, top: float | None, base: float | None) -> tuple[float, float]:
    """Validate an interval against the logged range and return the (trimmed) interval."""
    lo, hi = float(df.index.min()), float(df.index.max())
    unit = df.attrs.get("depth_unit") or ""
    top = lo if top is None else float(top)
    base = hi if base is None else float(base)
    if top > base:
        raise WellLogError(f"Top ({top:g}) must be shallower than base ({base:g}).")
    if base < lo or top > hi:
        raise WellLogError(
            f"Interval {top:g}–{base:g} {unit} is outside the logged depth range "
            f"{lo:g}–{hi:g} {unit}.".replace("  ", " ")
        )
    return max(top, lo), min(base, hi)


def list_curves(df: pd.DataFrame, curves: list[dict]) -> dict:
    """Describe the well: depth range, step and each curve with its valid-data extent."""
    df = _sorted(df)
    out = []
    for c in curves:
        col = c["mnemonic"]
        valid = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        out.append({
            **c,
            "valid_samples": int(valid.size),
            "first_valid_depth": _f(valid.index.min()) if valid.size else None,
            "last_valid_depth": _f(valid.index.max()) if valid.size else None,
        })
    return {
        "well": df.attrs.get("well"),
        "depth_unit": df.attrs.get("depth_unit"),
        "top": _f(df.index.min()),
        "base": _f(df.index.max()),
        "step": _step(df),
        "curves": out,
    }


def curve_stats(df: pd.DataFrame, curve: str, top: float | None = None,
                base: float | None = None) -> dict:
    """Mean, min, max, P10/P50/P90 and valid-sample count of ``curve`` over [top, base]."""
    df = _sorted(df)
    col = resolve_curve(df, curve)
    top, base = _interval(df, top, base)
    values = df.loc[top:base, col].dropna()
    if values.empty:
        raise WellLogError(f"{col} has no valid samples between {top:g} and {base:g}.")
    p10, p50, p90 = np.percentile(values.to_numpy(dtype=float), [10, 50, 90])
    return {
        "curve": col,
        "unit": _unit(df, col),
        "depth_unit": df.attrs.get("depth_unit"),
        "top": top,
        "base": base,
        "count": int(values.size),
        "mean": math.fsum(values) / values.size,  # fsum: no accumulated rounding error
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
    }


def flag_zones(df: pd.DataFrame, curve: str, threshold: float, above: bool = True,
               min_thickness: float = 0, top: float | None = None,
               base: float | None = None) -> list[dict]:
    """Find contiguous zones where ``curve`` is above (or below) ``threshold``.

    Each zone is ``{top, base, thickness, mean}``. ``top``/``base`` are the first and last
    flagged sample depths; ``thickness`` is samples × step, so a one-sample zone is one step
    thick. Missing values end a zone. Zones thinner than ``min_thickness`` are dropped.
    """
    df = _sorted(df)
    col = resolve_curve(df, curve)
    top, base = _interval(df, top, base)
    step = _step(df)
    values = df.loc[top:base, col]
    flagged = (values > threshold) if above else (values < threshold)  # NaN -> False

    run_id = (flagged != flagged.shift()).cumsum()
    zones = []
    for _, run in values[flagged].groupby(run_id[flagged]):
        thickness = run.size * step
        if thickness < min_thickness:
            continue
        zones.append({
            "top": float(run.index[0]),
            "base": float(run.index[-1]),
            "thickness": float(thickness),
            "mean": math.fsum(run) / run.size,
        })
    return zones
