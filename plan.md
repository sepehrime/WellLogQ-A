# Well Log Q&A Agent: Development Plan

A small Python project that loads a LAS file and answers plain-English questions about it, such as
_"What's the average porosity between 2,500 and 2,700 m?"_ or _"Flag zones with high gamma ray."_

**Stack:** Python · lasio · pandas · Plotly · Anthropic SDK (Claude) · Jupyter · pytest

**Core rule:** the LLM never does the math. Plain Python functions compute every number, and Claude only
chooses which function to call and explains the result.

Each phase has **Build** tasks and **Verify** checks. Finish all the Verify boxes before starting the next phase.

---

## Phase 0: Project setup

**Build**

- [x] Create the folder `WllLogQ-A/`
- [x] Create a virtual environment (`python -m venv .venv`) and activate it
- [x] Add `requirements.txt` listing lasio, pandas, numpy, plotly, anthropic, python-dotenv, jupyter, pytest
- [x] `pip install -r requirements.txt`
- [x] Create the folders `welllog/`, `data/`, `notebooks/`, `tests/` and an empty `welllog/__init__.py`
- [ ] Add a `.env` file containing `ANTHROPIC_API_KEY=...`

**Verify**

- [x] `python -c "import lasio, pandas, plotly, anthropic"` runs with no errors
- [x] `jupyter notebook` opens and a cell can `import welllog`

---

## Phase 1: Load the LAS file (`welllog/data.py`)

**Build**

- [ ] Put a sample LAS file in `data/`: one of yours, or a public one (Kansas Geological Survey, Equinor Volve)
- [ ] Write `load_las(path) -> (df, curves)`
  - [ ] `lasio.read(path).df()` gives a DataFrame indexed by depth
  - [ ] Replace null values (`-999.25`) with NaN
  - [ ] `curves` is a list of `{mnemonic, unit, description}` from the header
  - [ ] Detect the depth unit (m or ft)
- [ ] Notebook cell: load the file and show `df.head()`, `df.describe()` and the curve list

**Verify**

- [ ] The depth range and curve names match what the LAS header says
- [ ] No `-999.25` values are left (`(df == -999.25).sum().sum() == 0`)
- [ ] You know which curves are gamma ray and porosity, and their units (API; v/v or %)

---

## Phase 2: Analysis tools (`welllog/tools.py`)

**Build**

- [ ] `list_curves(df, curves)` returns mnemonics, units, descriptions and depth range
- [ ] `curve_stats(df, curve, top=None, base=None)` returns mean, min, max, P10/P50/P90 and the count of valid samples
- [ ] `flag_zones(df, curve, threshold, above=True, min_thickness=0)` returns `[{top, base, thickness, mean}]`
- [ ] Bad input gives a clear error message, not a crash (unknown curve, interval outside the logged depth, top > base)
- [ ] `tests/test_tools.py` uses a small made-up DataFrame with known values

**Verify**

- [ ] `pytest tests/` passes
- [ ] Test: constant porosity of 0.20 gives a mean of exactly 0.20
- [ ] Test: one GR spike gives exactly one flagged zone with the correct top and base
- [ ] On the real file, `curve_stats` matches a manual `df.loc[top:base, curve].mean()`

---

## Phase 3: Plotting (`plot_logs` in `welllog/tools.py`)

**Build**

- [ ] `plot_logs(df, curves, top=None, base=None, zones=None)` returns a Plotly figure
  - [ ] One track per curve: `make_subplots(rows=1, cols=n, shared_yaxes=True)`
  - [ ] The depth axis is reversed, so depth increases downward
  - [ ] Zones are drawn as shaded bands (`add_hrect`)
  - [ ] Axis labels include units

**Verify**

- [ ] The notebook shows GR and porosity tracks side by side
- [ ] High-GR zones from `flag_zones` appear as shaded bands in the right places
- [ ] Zooming or panning one track moves the others with it

---

## Phase 4: The agent (`welllog/agent.py`)

**Build**

- [ ] `WellLogAgent(las_path)` loads the data once and keeps `df` and `curves`
- [ ] Wrap each tool with `@beta_tool` (from `anthropic`). Each wrapper has a clear docstring and returns JSON
- [ ] `plot_logs` wrapper: store the figure in `agent.figures` and send Claude back only "plot created"
- [ ] Write a short system prompt: use tools for every number, state the curve and interval used, say which threshold was picked
- [ ] `ask(question) -> str` uses `client.beta.messages.tool_runner(model="claude-opus-5-5", max_tokens=16000, ...)`
- [ ] Check `stop_reason` before reading the answer (handle `"refusal"`)
- [ ] Load the API key from `.env` with `python-dotenv`

**Verify**

- [ ] `agent.ask("What curves are in this file?")` lists the real curves
- [ ] `agent.ask("Average porosity between 2500 and 2700 m?")` matches a direct `curve_stats(...)` call
- [ ] The answer names the curve used (for example `NPHI`) and the units
- [ ] Asking about an unknown curve or out-of-range depths gives a helpful reply, not a traceback

---

## Phase 5: Demo notebook (`notebooks/demo.ipynb`)

**Build**

- [ ] Section 1: load and preview the well
- [ ] Section 2: call the tools directly (no LLM)
- [ ] Section 3: agent questions
  - [ ] "What's the average porosity between 2,500 and 2,700 m?"
  - [ ] "Flag zones with high gamma ray" (show the answer and `agent.figures[-1].show()`)
  - [ ] One question of your own
- [ ] Short `README.md`: what it does, setup steps, how to run it

**Verify**

- [ ] "Restart kernel & run all" finishes with no errors
- [ ] The answers in Section 3 match the direct calls in Section 2
- [ ] Someone else could follow the README in a fresh folder

---

## Out of scope (on purpose)

No web UI, no multi-well support, no saved chat memory, and no petrophysical calculations beyond basic stats and thresholds.
Possible later additions: a Vsh calculation from GR, ft↔m conversion, a Streamlit front end, comparing several wells.
