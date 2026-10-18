# WellLogQ-A

AI can help you understand a well log. This project loads a LAS file, plots it, and lets you ask
plain-English questions about it, such as *"What's the average porosity between 2,500 and 2,700 ft?"*
or *"Flag zones with high gamma ray."*

**Core rule:** the LLM never does the math. Plain Python functions compute every number; Claude
(Opus 5.5, through [OpenRouter](https://openrouter.ai)) only chooses which function to call and
explains the result.

## What's inside

| Path | What it does |
|---|---|
| `welllog/data.py` | `load_las(path)` → depth-indexed DataFrame (nulls as NaN) + curve list |
| `welllog/tools.py` | `list_curves`, `curve_stats`, `flag_zones`, `plot_logs` (Plotly) |
| `welllog/agent.py` | `WellLogAgent`: Claude with the tools above |
| `notebooks/demo.ipynb` | End-to-end demo: load, direct tool calls, agent questions |
| `data/sample.las` | Sample well (Busse #1-27, Cheyenne County, Kansas; depths in **feet**) |

## Setup

Needs Python 3.10+ and an [OpenRouter API key](https://openrouter.ai/keys).

```bash
git clone <this repo> && cd WellLogQ-A
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo 'OPENROUTER_API_KEY=sk-or-...' > .env   # your key; .env is git-ignored
```

## Run it

**Demo notebook:**

```bash
jupyter notebook notebooks/demo.ipynb
```

Then *Kernel → Restart & Run All*. Sections 1–2 work without a key; Section 3 calls Claude.

**From Python** (run from the repo root):

```python
from welllog.agent import WellLogAgent

agent = WellLogAgent("data/sample.las")
print(agent.ask("What's the average porosity between 2500 and 2700 ft?"))
agent.last_calls          # the tool calls (inputs and outputs) behind that answer
agent.figures[-1].show()  # after a question that asks for a plot
```

The tools also work on their own, with no API key:

```python
from welllog.data import load_las
from welllog.tools import curve_stats, flag_zones, plot_logs

df, curves = load_las("data/sample.las")
curve_stats(df, "CNLS", top=2500, base=2700)                # mean, min, max, P10/P50/P90, count
zones = flag_zones(df, "GR", threshold=150, min_thickness=5)
plot_logs(df, ["GR", "CNLS"], zones=zones).show()
```

**Tests** (offline, no key needed):

```bash
pytest                                     # all tests
pytest tests/test_tools.py::test_one_gr_spike_gives_one_zone   # a single test
```

## Things to know

- Depths are always in the file's own unit (the sample is in feet). Nothing is converted between ft and m.
- Porosity curves in the sample (`CNLS`, `CNSS`, `CNDL`, `DPOR`) are in PU, i.e. percent.
- Each `ask` is independent: there is no chat memory between questions.
- Every `ask` is a paid API call on your OpenRouter account.
