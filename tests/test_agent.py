"""Offline tests for WellLogAgent: the tool wrappers and stop_reason handling (no API calls)."""

import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from welllog.agent import FREE_MODELS, WellLogAgent, untraced_numbers
from welllog.tools import curve_stats

SAMPLE = Path(__file__).parent.parent / "data" / "sample.las"


class FakeClient:
    """Stands in for anthropic.Anthropic: records the runner params, returns a canned message.

    ``errors`` maps a model slug to the exception its request raises.
    """

    def __init__(self, message, errors=None):
        self.params = None
        self.models_tried = []
        errors = errors or {}

        def tool_runner(**params):
            self.params = params
            self.models_tried.append(params["model"])
            if params["model"] in errors:
                raise errors[params["model"]]
            return SimpleNamespace(until_done=lambda: message)

        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=tool_runner))


def message(stop_reason, *texts, stop_details=None):
    content = [SimpleNamespace(type="text", text=t) for t in texts]
    return SimpleNamespace(stop_reason=stop_reason, content=content, stop_details=stop_details)


@pytest.fixture
def agent():
    return WellLogAgent(SAMPLE, client=FakeClient(message("end_turn", "ok")))


def tool(agent, name):
    return next(t for t in agent.tools if t.name == name)


def test_loads_data_once(agent):
    assert agent.df.shape == (9729, 22)
    assert len(agent.curves) == 22
    assert "Busse #1-27" in agent.system and "ft" in agent.system


def test_list_curves_tool(agent):
    info = json.loads(tool(agent, "list_curves").call({}))
    assert [c["mnemonic"] for c in info["curves"]] == list(agent.df.columns)


def test_curve_stats_tool_matches_direct_call(agent):
    out = json.loads(tool(agent, "curve_stats").call({"curve": "cnls", "top": 2500, "base": 2700}))
    assert out == curve_stats(agent.df, "CNLS", 2500, 2700)


def test_tool_errors_come_back_as_json(agent):
    out = json.loads(tool(agent, "curve_stats").call({"curve": "NPHI"}))
    assert out["error"].startswith("Unknown curve 'NPHI'")
    out = json.loads(tool(agent, "flag_zones").call({"curve": "GR", "threshold": 1, "top": 9000}))
    assert "outside the logged depth range" in out["error"]


def test_flag_zones_tool_summary(agent):
    out = json.loads(tool(agent, "flag_zones").call({"curve": "GR", "threshold": 150, "min_thickness": 5}))
    assert out["zone_count"] == 13 and len(out["zones"]) == 13 and not out["truncated"]
    out = json.loads(tool(agent, "flag_zones").call({"curve": "GR", "threshold": 50}))
    assert out["truncated"] and len(out["zones"]) == 50 and out["zone_count"] > 50


def test_plot_tool_stores_figure_and_returns_short_text(agent):
    out = tool(agent, "plot_logs").call(
        {"curves": ["GR", "CNLS"], "zone_curve": "GR", "zone_threshold": 150, "zone_min_thickness": 5})
    assert out == "plot created"
    assert len(agent.figures) == 1 and len(agent.figures[0].layout.shapes) == 13 * 2
    assert "error" in json.loads(tool(agent, "plot_logs").call({"curves": ["GR"], "zone_curve": "GR"}))
    assert len(agent.figures) == 1


def test_ask_returns_text_and_sends_expected_params():
    client = FakeClient(message("end_turn", "Mean CNLS is 28.7 PU."))
    agent = WellLogAgent(SAMPLE, client=client)
    assert agent.ask("porosity?") == "Mean CNLS is 28.7 PU."
    p = client.params
    assert p["model"] == FREE_MODELS[0] and agent.last_model == FREE_MODELS[0]
    assert p["max_tokens"] == 16000
    assert p["messages"] == [{"role": "user", "content": "porosity?"}]
    assert [t.name for t in p["tools"]] == ["list_curves", "curve_stats", "flag_zones", "plot_logs"]


def test_ask_handles_refusal():
    details = SimpleNamespace(explanation="policy", category=None)
    agent = WellLogAgent(SAMPLE, client=FakeClient(message("refusal", stop_details=details)))
    assert agent.ask("x") == "The model declined to answer this question. (policy)"


def test_ask_flags_truncation():
    agent = WellLogAgent(SAMPLE, client=FakeClient(message("max_tokens", "partial")))
    assert agent.ask("x").startswith("partial") and "cut off" in agent.ask("x")


def test_openrouter_client_uses_only_the_openrouter_key(monkeypatch):
    from welllog import agent as agent_mod
    monkeypatch.setattr(agent_mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-be-sent")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client = agent_mod.openrouter_client()
    assert str(client.base_url).rstrip("/") == "https://openrouter.ai/api"
    assert client.api_key is None and client.auth_token == "sk-or-test"
    assert client.max_retries == 1 and client.timeout == 180.0
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        agent_mod.openrouter_client()


def test_tool_calls_are_logged(agent):
    tool(agent, "curve_stats").call({"curve": "GR", "top": 1000, "base": 1100})
    tool(agent, "curve_stats").call({"curve": "NPHI"})
    assert [c["tool"] for c in agent.last_calls] == ["curve_stats", "curve_stats"]
    assert agent.last_calls[0]["input"] == {"curve": "GR", "top": 1000, "base": 1100}
    assert agent.last_calls[0]["output"]["count"] == 201
    assert "error" in agent.last_calls[1]["output"]
    agent.ask("anything")
    assert agent.last_calls == []


def api_error(cls, status, text=None):
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://openrouter.ai/api/v1/messages"))
    return cls(text or f"HTTP {status}", response=response, body=None)


def test_ask_falls_back_to_next_model_when_rate_limited():
    client = FakeClient(message("end_turn", "ok"), errors={
        "a:free": api_error(anthropic.RateLimitError, 429),
        "b:free": api_error(anthropic.PermissionDeniedError, 403),
    })
    agent = WellLogAgent(SAMPLE, client=client, models=["a:free", "b:free", "c:free"])
    assert agent.ask("x") == "ok"
    assert client.models_tried == ["a:free", "b:free", "c:free"]
    assert agent.last_model == "c:free"


def test_ask_raises_when_every_model_fails():
    errors = {m: api_error(anthropic.RateLimitError, 429) for m in ("a:free", "b:free")}
    agent = WellLogAgent(SAMPLE, client=FakeClient(message("end_turn", "ok"), errors), models=["a:free", "b:free"])
    with pytest.raises(RuntimeError, match="(?s)No model could answer.*a:free: 429.*b:free: 429"):
        agent.ask("x")


def test_bad_request_is_not_retried_on_other_models():
    client = FakeClient(message("end_turn", "ok"), errors={"a:free": api_error(anthropic.BadRequestError, 400)})
    agent = WellLogAgent(SAMPLE, client=client, models=["a:free", "b:free"])
    with pytest.raises(anthropic.BadRequestError):
        agent.ask("x")
    assert client.models_tried == ["a:free"]


def test_models_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "x/one:free, x/two:free")
    assert WellLogAgent(SAMPLE, client=FakeClient(None)).models == ["x/one:free", "x/two:free"]
    monkeypatch.delenv("OPENROUTER_MODEL")
    assert WellLogAgent(SAMPLE, client=FakeClient(None)).models == list(FREE_MODELS)
    assert WellLogAgent(SAMPLE, client=FakeClient(None), models="x/only").models == ["x/only"]


def test_ask_falls_back_on_connection_error_and_model_limit():
    request = httpx2.Request("POST", "https://openrouter.ai/api/v1/messages")
    client = FakeClient(message("end_turn", "ok"), errors={
        "a:free": anthropic.APITimeoutError(request=request),
        "b:free": api_error(anthropic.BadRequestError, 400, "max_tokens: 16000 > 8192, the maximum allowed"),
    })
    agent = WellLogAgent(SAMPLE, client=client, models=["a:free", "b:free", "c:free"])
    assert agent.ask("x") == "ok" and client.models_tried == ["a:free", "b:free", "c:free"]


def test_invalid_tool_arguments_are_logged_and_returned(agent):
    out = json.loads(tool(agent, "plot_logs").call({"curves": ["GR"], "zone_curve": "GR", "zone_threshold": "high"}))
    assert out["error"].startswith("Tool call failed. Invalid arguments for function plot_logs")
    assert agent.last_calls == [{"tool": "plot_logs", "output": out,
                                 "input": {"curves": ["GR"], "zone_curve": "GR", "zone_threshold": "high"}}]


def test_unexpected_tool_errors_are_logged(agent, monkeypatch):
    from welllog import tools as tools_mod

    def broken(*args, **kwargs):
        raise KeyError("boom")

    monkeypatch.setattr(tools_mod, "flag_zones", broken)
    out = json.loads(tool(agent, "flag_zones").call({"curve": "GR", "threshold": 1}))
    assert out["error"] == "Tool call failed. KeyError: 'boom'"
    assert [(c["tool"], c["output"]) for c in agent.last_calls] == [("flag_zones", out)]


def test_untraced_numbers():
    calls = [{"tool": "curve_stats", "input": {"curve": "CNLS", "top": 2500, "base": 2700},
              "output": {"mean": 28.7349, "count": 401, "p10": 20.04, "min": -1.126}}]
    well = "Loaded well: Busse #1-27, depth unit: ft, logged from 345 to 5209."
    ok = ("Mean CNLS over 2,500–2,700 ft is 28.73 PU (about 29), n = 401.\n"
          "1. P10 is 20.0 and the minimum is -1.13 PU, for the 10th percentile. Busse #1-27, top 345 ft.")
    assert untraced_numbers(ok, "between 2,500 and 2,700 ft?", well, calls) == []
    bad = "2,500–2,700 m is roughly 8,202–8,858 ft; mean 28.7 PU, 12.5 PU above average."
    assert untraced_numbers(bad, "2,500 to 2,700 m", well, calls) == ["8,202", "8,858", "12.5"]


def test_agent_untraced_numbers_uses_last_question_and_calls():
    agent = WellLogAgent(SAMPLE, client=FakeClient(message("end_turn", "GR mean is 86.5 GAPI from 345 to 5209 ft.")))
    assert agent.untraced_numbers() == []                    # nothing asked yet
    agent.ask("What is the mean GR?")
    assert agent.last_question == "What is the mean GR?"
    assert agent.untraced_numbers() == ["86.5"]               # no tool call backs it; 345/5209 come from well info
    tool(agent, "curve_stats").call({"curve": "GR"})          # mean 86.4545...
    assert agent.untraced_numbers() == []
