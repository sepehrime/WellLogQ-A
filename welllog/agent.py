"""LLM agent that answers questions about one LAS file using the tools in ``welllog.tools``.

The model never computes numbers itself: every value comes from a tool call, and the model
only picks the tool and explains the result.

Requests go through OpenRouter's Anthropic-compatible endpoint, authenticated with
``OPENROUTER_API_KEY`` (from the environment or ``.env``), using the Anthropic SDK. By default
it uses free OpenRouter models (``FREE_MODELS``), trying the next one when a model is
rate-limited or unavailable. Set ``OPENROUTER_MODEL`` (comma-separated) to choose others.

``untraced_numbers`` checks an answer for numbers that came from neither the question nor a
tool result, so a breach of the "no math in prose" rule is visible.
"""

import json
import math
import os
import re
from pathlib import Path

import anthropic
from anthropic import beta_tool
from dotenv import load_dotenv

from welllog import tools
from welllog.data import load_las
from welllog.tools import WellLogError

OPENROUTER_BASE_URL = "https://openrouter.ai/api"
# Free, tool-capable OpenRouter models, tried in order. Free models come and go: check
# https://openrouter.ai/models?max_price=0 if all of these stop working.
FREE_MODELS = (
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/free",  # OpenRouter's router: picks a random free model per request
)
# Errors that mean "this model can't serve us right now", so the next model is tried.
_TRY_NEXT_MODEL = (
    anthropic.RateLimitError,         # 429: free-tier or upstream rate limit
    anthropic.InternalServerError,    # 5xx: provider down or overloaded
    anthropic.NotFoundError,          # model removed from OpenRouter
    anthropic.PermissionDeniedError,  # model gated (some free models are app-only)
    anthropic.APIConnectionError,     # network failure or timeout (APITimeoutError)
)
# A 400 is normally our bug and is raised, unless it is about this model's own limits.
_MODEL_LIMIT_HINTS = ("max_tokens", "max tokens", "context length", "context window", "context_length")
REQUEST_TIMEOUT = 180.0  # seconds per request; the SDK default of 10 minutes stalls on a hung provider
MAX_RETRIES = 1          # per model; falling back to the next model beats waiting out backoff
MAX_ZONES_RETURNED = 50
MAX_ERROR_CHARS = 500

SYSTEM_PROMPT = """You answer questions about a single well log (LAS file) for a petroleum geologist.

Rules:
- Every number in your answer must appear in a tool result or in the user's question. Never compute, \
estimate, convert or recall values yourself: no unit conversions, differences, ratios or sums of your own.
- Always state which curve (mnemonic) and which depth interval you used, with units. Depths are in the \
well's depth unit. If the user gives depths in a different unit, say so, do not convert them, and ask \
them to restate the interval in the well's unit.
- When you pick a threshold yourself (for example "high gamma ray"), say what value you chose and why.
- If the user names a property rather than a curve (porosity, gamma ray, resistivity), call list_curves first, \
choose the matching curve, and say which one you chose if several fit.
- If a tool returns an error, explain it plainly and suggest a valid alternative (an available curve, \
the logged depth range).
- Note when the data looks unreliable: curve_stats reports suspect_count (samples outside the curve's \
usual valid_range, such as filler values or spikes), and count shows how many valid samples there are.
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
    return anthropic.Anthropic(base_url=OPENROUTER_BASE_URL, auth_token=key,
                               max_retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT)


def _is_model_limit(error: anthropic.BadRequestError) -> bool:
    text = str(getattr(error, "message", error)).lower()
    return any(hint in text for hint in _MODEL_LIMIT_HINTS)


# A number as written in prose: optional sign, optional thousands commas, optional decimals.
# Skips numbers glued to letters, "#" or "." (P10, #1, v2, 10th) and markdown list markers.
_NUMBER = re.compile(r"(?<![A-Za-z_#.\d])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d|st\b|nd\b|rd\b|th\b)")
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")


def _numbers_in_text(text: str) -> list[str]:
    return _NUMBER.findall(_LIST_MARKER.sub(" ", text))


def _collect_numbers(obj, out: set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.add(float(obj))
    elif isinstance(obj, str):
        out.update(float(n.replace(",", "")) for n in _numbers_in_text(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_numbers(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_numbers(value, out)


def untraced_numbers(answer: str, *sources) -> list[str]:
    """Numbers in ``answer`` that don't match any number in ``sources``.

    ``sources`` are strings (the question, the well description) and tool calls or results
    (dicts/lists, searched recursively). A number written with d decimals matches a source
    value that rounds to it at d decimals, so "28.7" matches 28.7349. An empty list means
    every number in the answer is traceable to the question or a tool.
    """
    known: set[float] = set()
    for source in sources:
        _collect_numbers(source, known)
    untraced = []
    for text in _numbers_in_text(answer):
        value = float(text.replace(",", ""))
        decimals = len(text.split(".")[1]) if "." in text else 0
        tolerance = 0.5 * 10 ** -decimals + 1e-9
        if not any(abs(value - k) <= tolerance for k in known) and text not in untraced:
            untraced.append(text)
    return untraced


def _default_models() -> list[str]:
    env = os.environ.get("OPENROUTER_MODEL", "")
    return [m.strip() for m in env.split(",") if m.strip()] or list(FREE_MODELS)


class WellLogAgent:
    """Load a LAS file once and answer plain-English questions about it with an LLM.

    Each ``ask`` is independent (no chat memory). Figures produced by the plotting
    tool are appended to ``self.figures``. ``self.last_calls`` lists every tool call made
    during the most recent ``ask`` as ``{tool, input, output}``, failed ones included, so
    answers can be checked; ``self.last_model`` names the model that answered, and
    ``untraced_numbers()`` lists numbers in the last answer that no tool or the question gave.

    ``models`` is an OpenRouter model slug or a list tried in order; it defaults to
    ``OPENROUTER_MODEL`` (comma-separated) or else ``FREE_MODELS``.
    """

    def __init__(self, las_path: str | Path, client: anthropic.Anthropic | None = None,
                 models: str | list[str] | None = None):
        self.df, self.curves = load_las(las_path)
        self.figures = []
        self.last_calls = []
        self.last_model = None
        self.last_question = self.last_answer = None
        self.client = client or openrouter_client()  # loads .env, so read OPENROUTER_MODEL after
        self.models = [models] if isinstance(models, str) else list(models or _default_models())
        self.tools = [self._guard(t) for t in self._make_tools()]
        unit = self.df.attrs.get("depth_unit") or "unknown unit"
        self.well_info = (
            f"Loaded well: {self.df.attrs.get('well')}, depth unit: {unit}, "
            f"logged from {self.df.index.min():g} to {self.df.index.max():g}."
        )
        self.system = f"{SYSTEM_PROMPT}\n\n{self.well_info}"

    def _log(self, tool: str, args: dict, out: str) -> str:
        try:
            output = json.loads(out)
        except ValueError:
            output = out
        self.last_calls.append({"tool": tool, "input": args, "output": output})
        return out

    def _guard(self, tool):
        """Make every failed call of ``tool`` show up in ``last_calls`` and reach the model as JSON.

        Covers arguments the SDK rejects before our function runs (wrong types) and unexpected
        errors inside it; ``WellLogError`` is already handled inside each wrapper.
        """
        call = tool.call

        def guarded(input):
            try:
                return call(input)
            except Exception as e:  # noqa: BLE001 - any tool failure must be logged and reported
                detail = f"{e}: {e.__cause__}" if e.__cause__ else f"{type(e).__name__}: {e}"
                args = dict(input) if isinstance(input, dict) else {"input": input}
                return self._log(tool.name, args, _json({"error": f"Tool call failed. {detail}"[:MAX_ERROR_CHARS]}))

        tool.call = guarded
        return tool

    def _make_tools(self) -> list:
        df, curves = self.df, self.curves
        log = self._log

        @beta_tool
        def list_curves() -> str:
            """List the well's curves with units and descriptions, plus the logged depth range,
            depth unit, sample step and each curve's valid-data extent. Call this first when you
            need to know which curve measures a property."""
            return log("list_curves", {}, _json(tools.list_curves(df, curves)))

        @beta_tool
        def curve_stats(curve: str, top: float | None = None, base: float | None = None) -> str:
            """Statistics for one curve over a depth interval: count of valid samples, mean, min,
            max and P10/P50/P90 (statistical percentiles), with units and the interval actually used.

            Args:
                curve: Curve mnemonic, e.g. GR or CNLS (case-insensitive).
                top: Shallowest depth of the interval, in the well's depth unit. Omit for the top of the log.
                base: Deepest depth of the interval, in the well's depth unit. Omit for the bottom of the log.
            """
            args = {"curve": curve, "top": top, "base": base}
            try:
                return log("curve_stats", args, _json(tools.curve_stats(df, curve, top, base)))
            except WellLogError as e:
                return log("curve_stats", args, _json({"error": str(e)}))

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
            args = {"curve": curve, "threshold": threshold, "above": above,
                    "min_thickness": min_thickness, "top": top, "base": base}
            try:
                zones = tools.flag_zones(df, curve, threshold, above, min_thickness, top, base)
            except WellLogError as e:
                return log("flag_zones", args, _json({"error": str(e)}))
            return log("flag_zones", args, _json({
                "curve": tools.resolve_curve(df, curve),
                "threshold": threshold,
                "above": above,
                "depth_unit": df.attrs.get("depth_unit"),
                "zone_count": len(zones),
                "total_thickness": sum(z["thickness"] for z in zones),
                "zones": zones[:MAX_ZONES_RETURNED],
                "truncated": len(zones) > MAX_ZONES_RETURNED,
            }))

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
            args = {"curves": curves, "top": top, "base": base, "zone_curve": zone_curve,
                    "zone_threshold": zone_threshold, "zone_above": zone_above,
                    "zone_min_thickness": zone_min_thickness}
            try:
                zones = None
                if zone_curve is not None:
                    if zone_threshold is None:
                        raise WellLogError("zone_threshold is required when zone_curve is given.")
                    zones = tools.flag_zones(df, zone_curve, zone_threshold, zone_above,
                                             zone_min_thickness, top, base)
                fig = tools.plot_logs(df, curves, top, base, zones)
            except WellLogError as e:
                return log("plot_logs", args, _json({"error": str(e)}))
            self.figures.append(fig)
            return log("plot_logs", args, "plot created")

        return [list_curves, curve_stats, flag_zones, plot_logs]

    def ask(self, question: str) -> str:
        """Answer one question. Returns the model's text answer, or an explanation if it stopped early.

        Tries ``self.models`` in order, moving on when a model is rate-limited or unavailable.
        Raises ``RuntimeError`` listing each model's error if none of them can answer.
        """
        failures = []
        self.last_question, self.last_answer = question, None
        for model in self.models:
            self.last_calls, n_figures = [], len(self.figures)
            try:
                message = self.client.beta.messages.tool_runner(
                    model=model,
                    max_tokens=16000,
                    system=self.system,
                    tools=self.tools,
                    messages=[{"role": "user", "content": question}],
                    max_iterations=20,
                ).until_done()
            except (*_TRY_NEXT_MODEL, anthropic.BadRequestError) as e:
                if isinstance(e, anthropic.BadRequestError) and not _is_model_limit(e):
                    raise
                del self.figures[n_figures:]  # drop plots from the abandoned attempt
                status = getattr(e, "status_code", type(e).__name__)
                failures.append(f"{model}: {status} {getattr(e, 'message', e)}")
                continue
            self.last_model = model
            self.last_answer = self._answer_text(message)
            return self.last_answer
        self.last_calls = []
        raise RuntimeError("No model could answer. Tried:\n" + "\n".join(failures))

    def untraced_numbers(self) -> list[str]:
        """Numbers in the last answer found in neither the question, the well info, nor any tool call."""
        if not self.last_answer:
            return []
        return untraced_numbers(self.last_answer, self.last_question, self.well_info, self.last_calls)

    @staticmethod
    def _answer_text(message) -> str:
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            reason = getattr(details, "explanation", None) if details else None
            return "The model declined to answer this question." + (f" ({reason})" if reason else "")

        text = "\n".join(b.text for b in message.content if b.type == "text").strip()
        if message.stop_reason == "max_tokens":
            text += "\n\n[Answer cut off: the response hit the token limit.]"
        return text or f"No answer was returned (stop reason: {message.stop_reason})."
