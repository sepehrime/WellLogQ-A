"""Analysis tools over a depth-indexed well-log DataFrame (as returned by ``load_las``).

Every function returns plain Python types (JSON-safe). Bad input raises ``WellLogError``
with a message meant to be shown to the user or the LLM.

Conventions:
  - ``top``/``base`` are inclusive depths in the file's depth unit; top must be shallower.
    An interval that only partly overlaps the logged range is trimmed to it, and the
    interval actually used is reported back.
  - P10/P50/P90 are the 10th/50th/90th statistical percentiles (P10 is the low value).
  - Values outside a curve's usual range (``valid_range``) are counted as suspect, not removed.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class WellLogError(ValueError):
    """Invalid request against the well data (unknown curve, bad interval, no data)."""


# Loose bounds on what a curve can plausibly read, used to count suspect samples (filler
# values such as 100000 ohm-m, tool spikes). Matched by mnemonic first, then by unit; curves
# with neither (for example corrections such as RHOC) get no range. Bounds are inclusive.
_RANGES_BY_CURVE = {
    "RHOB": (1.0, 3.5), "RHOZ": (1.0, 3.5),  # bulk density, g/cc
    "PE": (0.0, 20.0), "PEF": (0.0, 20.0),   # photoelectric factor, b/e
}
_RANGES_BY_UNIT = {
    "PU": (-30.0, 100.0),                      # porosity in percent (density porosity can go negative)
    "V/V": (-0.3, 1.0), "DEC": (-0.3, 1.0),    # porosity as a fraction
    "GAPI": (0.0, 500.0), "API": (0.0, 500.0),
    "OHM-M": (0.0, 10000.0), "OHMM": (0.0, 10000.0), "OHM.M": (0.0, 10000.0),
    "IN": (0.0, 40.0),                         # caliper
}


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


def _numeric(df: pd.DataFrame, curve: str) -> str:
    """Resolve ``curve`` and check that it holds numbers."""
    col = resolve_curve(df, curve)
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise WellLogError(f"{col} is not a numeric curve, so it has no statistics or zones.")
    return col


def valid_range(df: pd.DataFrame, curve: str) -> tuple[float, float] | None:
    """Plausible (low, high) bounds for ``curve``, or None if unknown."""
    col = resolve_curve(df, curve)
    return _RANGES_BY_CURVE.get(col.upper()) or _RANGES_BY_UNIT.get(_unit(df, col).strip().upper())


def _interval(df: pd.DataFrame, top: float | None, base: float | None) -> tuple[float, float]:
    """Validate an interval against the logged range and return the (trimmed) interval."""
    lo, hi = float(df.index.min()), float(df.index.max())
    unit = df.attrs.get("depth_unit") or ""
    top = lo if top is None else float(top)
    base = hi if base is None else float(base)
    if base < lo or top > hi:
        raise WellLogError(
            f"Interval {top:g}–{base:g} {unit} is outside the logged depth range "
            f"{lo:g}–{hi:g} {unit}.".replace("  ", " ")
        )
    if top > base:
        raise WellLogError(f"Top ({top:g}) must be shallower than base ({base:g}).")
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
    """Mean, min, max, P10/P50/P90 and valid-sample count of ``curve`` over [top, base].

    Also returns ``valid_range`` (plausible bounds, or None) and ``suspect_count``, the number
    of samples outside it (None when the range is unknown). Suspect samples stay in the stats.
    """
    df = _sorted(df)
    col = _numeric(df, curve)
    top, base = _interval(df, top, base)
    values = df.loc[top:base, col].dropna()
    if values.empty:
        raise WellLogError(f"{col} has no valid samples between {top:g} and {base:g}.")
    p10, p50, p90 = np.percentile(values.to_numpy(dtype=float), [10, 50, 90])
    bounds = valid_range(df, col)
    suspect = None if bounds is None else int(((values < bounds[0]) | (values > bounds[1])).sum())
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
        "valid_range": list(bounds) if bounds else None,
        "suspect_count": suspect,
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
    col = _numeric(df, curve)
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


def plot_logs(df: pd.DataFrame, curves: str | list[str], top: float | None = None,
              base: float | None = None, zones: list[dict] | None = None) -> go.Figure:
    """Plot one track per curve side by side, depth increasing downward.

    Tracks share the depth axis, so zooming or panning one moves the others. ``zones``
    (e.g. from ``flag_zones``) are drawn as shaded bands across all tracks, padded by half
    a step on each side so one-sample zones stay visible. Resistivity curves (ohm-m) use
    a log scale.
    """
    df = _sorted(df)
    if isinstance(curves, str):
        curves = [curves]
    if not curves:
        raise WellLogError("Give at least one curve to plot.")
    cols = [_numeric(df, c) for c in curves]
    top, base = _interval(df, top, base)
    data = df.loc[top:base, cols]
    depth_unit = df.attrs.get("depth_unit") or ""
    half = _step(df) / 2

    fig = make_subplots(rows=1, cols=len(cols), shared_yaxes=True, horizontal_spacing=0.02)
    for i, col in enumerate(cols, start=1):
        unit = _unit(df, col)
        fig.add_trace(
            go.Scatter(x=data[col], y=data.index, mode="lines", name=col, line={"width": 1},
                       hovertemplate=f"{col}: %{{x:.4g}} {unit}<br>Depth: %{{y}} {depth_unit}<extra></extra>"),
            row=1, col=i,
        )
        fig.update_xaxes(title_text=f"{col} ({unit})" if unit else col, side="top",
                         type="log" if "OHM" in unit.upper() else "linear", row=1, col=i)

    # One band per zone per track, set in a single call: add_hrect per zone gets very slow
    # with hundreds of zones.
    axes = [("x domain", "y")] + [(f"x{i} domain", f"y{i}") for i in range(2, len(cols) + 1)]
    fig.update_layout(shapes=[
        {"type": "rect", "xref": xref, "yref": yref, "x0": 0, "x1": 1,
         "y0": z["top"] - half, "y1": z["base"] + half, "fillcolor": "orange", "opacity": 0.3,
         "line": {"width": 0}, "layer": "below"}
        for z in zones or [] for xref, yref in axes
    ])

    fig.update_yaxes(range=[base, top], title_text=f"Depth ({depth_unit})" if depth_unit else "Depth",
                     row=1, col=1)
    title = df.attrs.get("well") or ""
    fig.update_layout(title=title, height=800, width=max(300, 220 * len(cols)), showlegend=False,
                      hovermode="y unified", margin={"t": 100})
    return fig
