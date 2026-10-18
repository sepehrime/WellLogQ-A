"""Offline tests for WellLogAgent: the tool wrappers and stop_reason handling (no API calls)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from welllog.agent import WellLogAgent
from welllog.tools import curve_stats

SAMPLE = Path(__file__).parent.parent / "data" / "sample.las"


class FakeClient:
    """Stands in for anthropic.Anthropic: records the runner params, returns a canned message."""

    def __init__(self, message):
        self.params = None
        runner = SimpleNamespace(until_done=lambda: message)

        def tool_runner(**params):
            self.params = params
            return runner

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
    assert p["model"] == "anthropic/claude-opus-5.5" and p["max_tokens"] == 16000
    assert p["messages"] == [{"role": "user", "content": "porosity?"}]
    assert [t.name for t in p["tools"]] == ["list_curves", "curve_stats", "flag_zones", "plot_logs"]


def test_ask_handles_refusal():
    details = SimpleNamespace(explanation="policy", category=None)
    agent = WellLogAgent(SAMPLE, client=FakeClient(message("refusal", stop_details=details)))
    assert agent.ask("x") == "Claude declined to answer this question. (policy)"


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
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        agent_mod.openrouter_client()
