"""Claude agent that answers questions about one LAS file using the tools in ``welllog.tools``.

Claude never computes numbers itself: every value comes from a tool call, and Claude only
picks the tool and explains the result.

Requests go through OpenRouter's Anthropic-compatible endpoint, authenticated with
``OPENROUTER_API_KEY`` (from the environment or ``.env``), using the Anthropic SDK.
"""

import json
import os
from pathlib import Path

import anthropic
from anthropic import beta_tool
from dotenv import load_dotenv

from welllog import tools
from welllog.data import load_las
from welllog.tools import WellLogError

OPENROUTER_BASE_URL = "https://openrouter.ai/api"
MODEL = "anthropic/claude-opus-5.5"  # OpenRouter model slug
MAX_ZONES_RETURNED = 50

SYSTEM_PROMPT = """You answer questions about a single well log (LAS file) for a petroleum geologist.

Rules:
- Every number in your answer must come from a tool result. Never compute, estimate or recall values yourself.
- Always state which curve (mnemonic) and which depth interval you used, with units. Depths are in the \
well's depth unit; if the user gives depths in a different unit, say so and do not silently convert.
- When you pick a threshold yourself (for example "high gamma ray"), say what value you chose and why.
- If the user names a property rather than a curve (porosity, gamma ray, resistivity), call list_curves first, \
choose the matching curve, and say which one you chose if several fit.
- If a tool returns an error, explain it plainly and suggest a valid alternative (an available curve, \
the logged depth range).
- Note when the data looks unreliable (impossible values, few valid samples).
- Be concise."""


def _json(obj) -> str:
    return json.dumps(obj, default=str)


def openrouter_client() -> anthropic.Anthropic:
    """Anthropic SDK client pointed at OpenRouter, using OPENROUTER_API_KEY from env or .env."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env: OPENROUTER_API_KEY=sk-or-...")
    # Passing auth_token explicitly stops the SDK from reading ANTHROPIC_* credentials,
    # so only the OpenRouter key is sent (as a Bearer token).
    return anthropic.Anthropic(base_url=OPENROUTER_BASE_URL, auth_token=key)


class WellLogAgent:
    """Load a LAS file once and answer plain-English questions about it with Claude.

    Each ``ask`` is independent (no chat memory). Figures produced by the plotting
    tool are appended to ``self.figures``.
    """

    def __init__(self, las_path: str | Path, client: anthropic.Anthropic | None = None,
                 model: str = MODEL):
        self.df, self.curves = load_las(las_path)
        self.figures = []
        self.model = model
        self.client = client or openrouter_client()
        self.tools = self._make_tools()
        unit = self.df.attrs.get("depth_unit") or "unknown unit"
        self.system = (
            f"{SYSTEM_PROMPT}\n\nLoaded well: {self.df.attrs.get('well')}, depth unit: {unit}, "
            f"logged from {self.df.index.min():g} to {self.df.index.max():g}."
        )

    def _make_tools(self) -> list:
        df, curves = self.df, self.curves

        @beta_tool
        def list_curves() -> str:
            """List the well's curves with units and descriptions, plus the logged depth range,
            depth unit, sample step and each curve's valid-data extent. Call this first when you
            need to know which curve measures a property."""
            return _json(tools.list_curves(df, curves))

        @beta_tool
        def curve_stats(curve: str, top: float | None = None, base: float | None = None) -> str:
            """Statistics for one curve over a depth interval: count of valid samples, mean, min,
            max and P10/P50/P90 (statistical percentiles), with units and the interval actually used.

            Args:
                curve: Curve mnemonic, e.g. GR or CNLS (case-insensitive).
                top: Shallowest depth of the interval, in the well's depth unit. Omit for the top of the log.
                base: Deepest depth of the interval, in the well's depth unit. Omit for the bottom of the log.
            """
            try:
                return _json(tools.curve_stats(df, curve, top, base))
            except WellLogError as e:
                return _json({"error": str(e)})

        @beta_tool
        def flag_zones(curve: str, threshold: float, above: bool = True, min_thickness: float = 0,
                       top: float | None = None, base: float | None = None) -> str:
            """Find contiguous depth zones where a curve is above (or below) a threshold.
            Returns the number of zones, their total thickness, and each zone's top, base,
            thickness and mean value (at most 50 zones listed).

            Args:
                curve: Curve mnemonic, e.g. GR (case-insensitive).
                threshold: Value to compare against, in the curve's units.
                above: True to flag values above the threshold, False for values below it.
                min_thickness: Drop zones thinner than this, in the well's depth unit.
                top: Shallowest depth to search. Omit for the top of the log.
                base: Deepest depth to search. Omit for the bottom of the log.
            """
            try:
                zones = tools.flag_zones(df, curve, threshold, above, min_thickness, top, base)
            except WellLogError as e:
                return _json({"error": str(e)})
            return _json({
                "curve": tools.resolve_curve(df, curve),
                "threshold": threshold,
                "above": above,
                "depth_unit": df.attrs.get("depth_unit"),
                "zone_count": len(zones),
                "total_thickness": sum(z["thickness"] for z in zones),
                "zones": zones[:MAX_ZONES_RETURNED],
                "truncated": len(zones) > MAX_ZONES_RETURNED,
            })

        @beta_tool
        def plot_logs(curves: list[str], top: float | None = None, base: float | None = None,
                      zone_curve: str | None = None, zone_threshold: float | None = None,
                      zone_above: bool = True, zone_min_thickness: float = 0) -> str:
            """Plot curves as side-by-side tracks for the user to see. Optionally shade zones where
            zone_curve is above/below zone_threshold (same rules as flag_zones). The plot is shown
            to the user directly; you only get a confirmation back.

            Args:
                curves: Curve mnemonics to plot, one track each, e.g. ["GR", "CNLS"].
                top: Shallowest depth to show. Omit for the top of the log.
                base: Deepest depth to show. Omit for the bottom of the log.
                zone_curve: Curve to flag zones on, e.g. GR. Omit for no zones.
                zone_threshold: Threshold for zone_curve, in its units. Required with zone_curve.
                zone_above: True to shade values above the threshold, False for below.
                zone_min_thickness: Drop zones thinner than this, in the well's depth unit.
            """
            try:
                zones = None
                if zone_curve is not None:
                    if zone_threshold is None:
                        raise WellLogError("zone_threshold is required when zone_curve is given.")
                    zones = tools.flag_zones(df, zone_curve, zone_threshold, zone_above,
                                             zone_min_thickness, top, base)
                fig = tools.plot_logs(df, curves, top, base, zones)
            except WellLogError as e:
                return _json({"error": str(e)})
            self.figures.append(fig)
            return "plot created"

        return [list_curves, curve_stats, flag_zones, plot_logs]

    def ask(self, question: str) -> str:
        """Answer one question. Returns Claude's text answer, or an explanation if it stopped early."""
        runner = self.client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=16000,
            system=self.system,
            tools=self.tools,
            messages=[{"role": "user", "content": question}],
            max_iterations=20,
        )
        message = runner.until_done()

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            reason = getattr(details, "explanation", None) if details else None
            return "Claude declined to answer this question." + (f" ({reason})" if reason else "")

        text = "\n".join(b.text for b in message.content if b.type == "text").strip()
        if message.stop_reason == "max_tokens":
            text += "\n\n[Answer cut off: the response hit the token limit.]"
        return text or f"No answer was returned (stop reason: {message.stop_reason})."
