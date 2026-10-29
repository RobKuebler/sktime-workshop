"""Builds the sktime workshop notebooks.

    python build/build_notebooks.py

Writes two notebooks into the project root:
  - sktime_workshop_instructor.ipynb  (solutions + instructor notes)
  - sktime_workshop.ipynb             (participants: exercise stubs, no solutions, no notes)

This file is the single source of truth. Edit here, never in the .ipynb files.
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent

# (kind, source) with kind in: md, code, notes (instructor only),
# task (exercise text, both), stub (participants only), solution (instructor only),
# time (timing checkpoint, instructor only)
CELLS: list[tuple[str, str]] = []


def md(s):
    CELLS.append(("md", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


def notes(s):
    CELLS.append(("notes", "💬 **Instructor notes**\n\n" + s.strip("\n")))


def timing(s):
    CELLS.append(("time", "⏱️ **Instructor timing**\n\n" + s.strip("\n")))


def exercise(title, task, stub, solution, bonus=False):
    label = "🌟 Bonus" if bonus else "✏️ Exercise"
    CELLS.append(("task", f"### {label}: {title}\n\n{task.strip()}"))
    CELLS.append(("stub", stub.strip("\n")))
    CELLS.append(("solution", "# --- Solution (instructor version only) ---\n" + solution.strip("\n")))


# =====================================================================
# 0. INTRO
# =====================================================================
md("""
# Forecasting with `sktime` — a hands-on workshop

**Sovereign AI Meetup Ulm · 17 November 2026**

You will learn how to forecast time series with [`sktime`](https://www.sktime.net), an open-source library with **one unified interface** for classical statistics, machine learning, and foundation models.

| Block | Topic | Time |
|---|---|---|
| 0 | Setup check | 20 min |
| 1 | Basic forecasting: `fit` / `predict`, evaluation, backtesting, ML models | 86 min |
| – | Break | 15 min |
| 2 | Pipelines, tuning, prediction intervals | 57 min |
| 3 | Many series at once: panel and hierarchical data | 32 min |
| 4 | Foundation models as drop-in forecasters | 37 min |
| – | Wrap-up | 5 min |

**How this works:** short explanations, live demos, and **exercises** (✏️) for you. Cells with `...` are yours to fill in. Bonus exercises (🌟) are for when you are faster than the rest.
""")

md("""
## What is `sktime`?

`sktime` is an open-source Python library for machine learning with time series. If you know `scikit-learn`, you already know the core idea:

1. **Pick** a model and set its parameters.
2. **`fit`** it on your data.
3. **`predict`**.

Want a different model? Change line 1. Lines 2 and 3 stay the same. That is the whole trick, and it works for ARIMA, random forests, and neural networks alike.

Time series are more than forecasting, and `sktime` covers all of it: **forecasting**, **classification**, **regression**, **clustering**, and **detection** of anomalies and changepoints. Today we only do forecasting, but everything you learn carries over.

Why does this matter? The Python time series world is a zoo. Every package has its own interface, and combining them is painful. `sktime` wraps many of them (statsmodels, pytorch-forecasting, Chronos, ...) behind one interface, so you can **swap, combine, and compare** models without rewriting your code. You will see this in Block 4, where a foundation model drops in just like `NaiveForecaster`.

`sktime` is community-driven and open source, with no vendor lock-in. Contributors are welcome, also beginners: [Discord](https://discord.com/invite/54ACzaFsn7) · [GitHub](https://github.com/sktime/sktime) · [Docs](https://www.sktime.net)
""")

notes("""
The timeline in the ⏱️ cells covers **everything** (full path: 252 min, ends at 4:12). For a **4-hour** workshop, skip the sections marked **SKIPPABLE** when you run behind: 1.5 exogenous variables (-10), the 2.3 exercise (-5), 4.3 context (-10), conformal intervals in 4.4 (-3). Skipping the first two gets you to 3:57. Both 🌟 bonus exercises are optional and not in the timeline. Compute is not the bottleneck (the whole notebook runs in ~2 min); talking, typing and troubleshooting are.

Rule for exercises: give 5-8 min, walk around, then show the solution. Every exercise is **self-contained**: later cells never use variables created in an exercise, so people who got stuck can still follow along.

Red thread of the day: *"Which model is best? Don't believe, measure (backtest)."* It starts in Block 1 (naive vs ETS), continues in Block 2 (a plain random forest loses, a pipeline wins) and ends in Block 4 (a foundation model is not automatically better).
""")

timing("""
**0:00-0:20 · Intro and Block 0 (20 min).** Intro talk 5 min, setup check 15 min: with a full room, someone always has a broken environment or a slow download. Do not wait for everybody, help stragglers during the next section.
""")

md("## Block 0 · Setup check (20 min)")

code("""
import logging
import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"  # also for worker processes
os.environ["HF_HUB_VERBOSITY"] = "error"  # no Hugging Face hints
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")  # keep the output clean for the workshop
warnings.filterwarnings("ignore", category=FutureWarning)  # sktime announces future changes, not relevant today
warnings.showwarning = lambda *args, **kwargs: None  # some libraries re-enable warnings on import
warnings.warn = lambda *args, **kwargs: None  # last resort: some libraries reset the filters themselves

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sktime

print("python ", sys.version.split()[0])
print("sktime ", sktime.__version__)
print("pandas ", pd.__version__)

plt.rcParams["figure.figsize"] = (10, 3.5)
C_BLUE, C_ORANGE = "#1f6fb4", "#e07b00"
""")

md("""
**Block 4 uses a pre-trained foundation model** that is downloaded once (about 35 MB) from Hugging Face. The next cell tests this now, so we find problems early.
""")

code("""
# Test that the foundation model of Block 4 can be loaded.
# This needs internet once; afterwards the model is cached on your disk.
from sktime.datasets import load_airline
from sktime.forecasting.chronos import ChronosForecaster

CHRONOS_OK = False
try:
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    ChronosForecaster("amazon/chronos-bolt-tiny").fit(load_airline(), fh=[1, 2])
    CHRONOS_OK = True
    print("✅ Chronos is ready.")
except Exception as e:  # noqa: BLE001
    print("⚠️ Could not load Chronos (no internet?). Blocks 1-3 work without it.")
    print("   In Block 4, the first demo has an offline fallback.")
    print("   Error:", type(e).__name__, str(e)[:150])
""")

notes("""
Ask for a show of hands: who sees ✅? Fix the rest while others read on. If a group has no internet: blocks 1-3 run offline, Block 4 section 4.1 has a fallback with stored predictions, the remaining Block 4 cells are skipped without the model.

Intro talking points (5 min): what is a forecast vs. a prediction in sklearn? Time order matters (no shuffling!). `sktime` = scikit-learn-like API for time series; same `fit`/`predict` for >100 forecasters.
""")

# =====================================================================
# 1. BASIC FORECASTING
# =====================================================================
timing("""
**0:20-0:25 · 1.1 the data (5 min).**
""")

md("""
---
# Block 1 · Basic forecasting (86 min)

## 1.1 The data

The classic *airline passengers* dataset: monthly international airline passengers 1949-1960. It has a **trend** and a **seasonality that grows with the level**.
""")

code("""
from sktime.datasets import load_airline

y = load_airline()
y.head()
""")

code("""
y.plot(title="Airline passengers (monthly)");
""")

md("""
`sktime` works with plain `pandas` objects. The index here is a `PeriodIndex` with monthly frequency. A `DatetimeIndex` or an integer index would work too.
""")

timing("""
**0:25-0:43 · 1.2 fit/predict + swap exercise (18 min).** Exercise 6 min including the solution. The registry cell is a 1-min aside.
""")

md("""
## 1.2 `fit` and `predict`

Every forecaster in `sktime` follows the same recipe:

1. **specify** the algorithm
2. **specify** the forecasting horizon `fh` (which steps do we want?)
3. **`fit`** on the history
4. **`predict`** the future

Let's start with the simplest forecaster: *repeat what happened one year ago* (seasonal naive).
""")

code("""
from sktime.forecasting.naive import NaiveForecaster

# step 1: the algorithm. "last" value, with a season of length 12 -> same month last year
naive = NaiveForecaster(strategy="last", sp=12)

# step 2: the horizon: the next 24 months
fh = np.arange(1, 25)

# step 3 and 4
naive.fit(y, fh=fh)
y_pred = naive.predict()

y_pred.head(3)
""")

code("""
from sktime.utils.plotting import plot_series

plot_series(y, y_pred, labels=["History", "Seasonal naive"]);
""")

notes("""
Stress: `fh=np.arange(1, 25)` means 1..24 steps ahead, *relative* to the end of the data. Absolute dates work too (`ForecastingHorizon`), but relative is the default for beginners.

Question for the audience: "Is this a good forecast?" (Looks plausible but ignores the trend.)
""")

exercise(
    "Swap the forecaster",
    """
Use **exponential smoothing** (ETS) instead of the naive forecaster. Same recipe, only step 1 changes.

* Create `ExponentialSmoothing(trend="add", seasonal="mul", sp=12)`
* `fit` it on `y`, `predict` the horizon `fh`
* plot history and forecast with `plot_series`
""",
    '''
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

ets = ...
ets.fit(...)
y_pred_ets = ...

plot_series(...)
''',
    '''
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

ets = ExponentialSmoothing(trend="add", seasonal="mul", sp=12)
ets.fit(y, fh=fh)
y_pred_ets = ets.predict()

plot_series(y, y_pred_ets, labels=["History", "ETS"]);
''',
)

md("""
Same methods, completely different algorithm. That is the point of `sktime`: **you can swap models without rewriting your code.**

*(If you skipped the exercise, run this cell so the rest works.)*
""")

code("""
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

ets = ExponentialSmoothing(trend="add", seasonal="mul", sp=12)
""")

md("""
How many forecasters are there? The registry knows:
""")

code("""
from sktime.registry import all_estimators

forecasters = all_estimators(estimator_types="forecaster", as_dataframe=True)
print(len(forecasters), "forecasters")
forecasters.head(10)
""")

md("""
Looking for a specific model? Try the **estimator search** on the website: <https://www.sktime.net/en/latest/estimator_overview.html>. You can filter by task and by capabilities, such as prediction intervals or exogenous data.
""")

timing("""
**0:43-0:53 · 1.3 train/test split and metrics (10 min).**
""")

md("""
## 1.3 How good is a forecast?

To judge a forecast, we hide the last part of the data (**test set**), forecast it, and compare. Unlike in scikit-learn, we must **not shuffle**: the test set is always the *end* of the series.

<img src="imgs/split.png" style="max-width: 900px" />
""")

code("""
from sktime.split import temporal_train_test_split

y_train, y_test = temporal_train_test_split(y, test_size=24)

naive.fit(y_train, fh=fh)
ets.fit(y_train, fh=fh)

plot_series(
    y_train.tail(36),
    y_test,
    naive.predict(),
    ets.predict(),
    labels=["Train", "Test", "Naive", "ETS"],
);
""")

code("""
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
)

mae = MeanAbsoluteError()
mape = MeanAbsolutePercentageError()

for name, model in [("Naive", naive), ("ETS", ets)]:
    y_hat = model.predict()
    print(f"{name:6} MAE: {mae(y_test, y_hat):6.1f}   MAPE: {mape(y_test, y_hat):.1%}")
""")

timing("""
**0:53-1:18 · 1.4 backtesting + exercise (25 min).** The splitter plot and the `backtest` helper need time to land. Exercise 6 min. ✅ Checkpoint 1:18: the `backtest` helper must be defined, it is used all day.
""")

md("""
## 1.4 Backtesting: one test set is not enough

A single test set can be lucky or unlucky. **Backtesting** (also: time series cross-validation) repeats the experiment on several windows: fit on the past, forecast the next 12 months, move the window forward, repeat.

We define the splitter **once** and use it for the whole workshop.

<img src="imgs/backtest.gif" style="max-width: 800px" />
""")

code("""
from sktime.split import ExpandingWindowSplitter
from sktime.utils.plotting import plot_windows

cv = ExpandingWindowSplitter(fh=np.arange(1, 13), initial_window=72, step_length=12)

plot_windows(cv=cv, y=y);
""")

code("""
from sktime.forecasting.model_evaluation import evaluate

results = evaluate(forecaster=ets, cv=cv, y=y, scoring=mape)
results[["test_MeanAbsolutePercentageError", "fit_time"]]
""")

md("""
One row per window. Let's wrap this in a small helper that returns the **average error over all windows**. We will reuse it all day.
""")

code("""
def backtest(forecaster, y=y, cv=cv):
    \"\"\"Mean MAPE over all backtest windows (lower is better).\"\"\"
    res = evaluate(forecaster=forecaster, cv=cv, y=y, scoring=mape)
    return float(res["test_MeanAbsolutePercentageError"].mean())


backtest(naive), backtest(ets)
""")

exercise(
    "Which forecaster wins?",
    """
Backtest **three** forecasters with the `backtest` helper and print them sorted by error (best first):

* `NaiveForecaster(strategy="last", sp=12)`
* `ExponentialSmoothing(trend="add", seasonal="mul", sp=12)`
* `ThetaForecaster(sp=12)` (from `sktime.forecasting.theta`)

*Hint: put them in a dictionary and loop over it.*
""",
    '''
from sktime.forecasting.theta import ThetaForecaster

candidates = {
    "naive": ...,
    "ets": ...,
    "theta": ...,
}

scores = {name: ... for name, f in candidates.items()}
pd.Series(scores).sort_values()
''',
    '''
from sktime.forecasting.theta import ThetaForecaster

candidates = {
    "naive": NaiveForecaster(strategy="last", sp=12),
    "ets": ExponentialSmoothing(trend="add", seasonal="mul", sp=12),
    "theta": ThetaForecaster(sp=12),
}

scores = {name: backtest(f) for name, f in candidates.items()}
pd.Series(scores).sort_values()
''',
)

notes("""
Expected (MAPE): ETS ~4%, Theta ~6%, Naive ~11%. Takeaway: a smarter model is not automatically better, but here ETS clearly beats the seasonal naive baseline. **Always keep the naive forecaster as a baseline.**

For fast people: `AutoETS(auto=True, sp=12)` or `AutoARIMA(sp=12)` as additional candidates.
""")

timing("""
**1:18-1:28 · 1.5 exogenous variables (10 min). ⏭️ SKIPPABLE (-10 min).** Nothing later depends on it (`y_lg`, `ForecastX` are only used here). If you skip it, subtract 10 min from all following times.
""")

md("""
## 1.5 Exogenous variables

Sometimes we know more than the history: weather, prices, promotions. In `sktime`, such **exogenous variables** are passed as `X`. Example: *US employment* (`y`) and macro-economic indicators (`X`).
""")

code("""
from sktime.datasets import load_longley

y_lg, X_lg = load_longley()
X_lg.head(3)
""")

code("""
from sktime.forecasting.arima import StatsModelsARIMA

y_lg_train, y_lg_test, X_lg_train, X_lg_test = temporal_train_test_split(
    y_lg, X_lg, test_size=4
)
fh_lg = np.arange(1, 5)

arima = StatsModelsARIMA(order=(1, 1, 0)).fit(y_lg_train, fh=fh_lg)
arimax = StatsModelsARIMA(order=(1, 1, 0)).fit(y_lg_train, X=X_lg_train, fh=fh_lg)

plot_series(
    y_lg_train,
    y_lg_test,
    arima.predict(),
    arimax.predict(X=X_lg_test),
    labels=["Train", "Test", "ARIMA", "ARIMA + X"],
);
""")

md("""
With `X`, the forecast looks better. But wait: in this evaluation we **knew the true values of `X` in the test period**. In real life, we want to forecast the *future*, and nobody gives us `X` for next year. Let's do it for real: fit on **all** data and predict the next 4 years.
""")

code("""
arimax_full = StatsModelsARIMA(order=(1, 1, 0)).fit(y_lg, X=X_lg, fh=fh_lg)

try:
    arimax_full.predict()  # no future X provided
except ValueError as e:
    print("❌ ValueError:", e)
""")

md("""
💥 **It fails.** To forecast `y`, the model needs the **future values of `X`**, and we don't have them. This is a very common trap with exogenous variables: *if you cannot know `X` in the future, you cannot use it directly.*

The fix: **forecast `X` as well.** That is what `ForecastX` does. It wraps two forecasters:

* `forecaster_X` forecasts every column of `X` into the future (**a second forecast!**)
* `forecaster_y` then uses these forecasted values as the future `X`

<img src="imgs/forecastx.gif" style="max-width: 700px" />
""")

code("""
from sktime.forecasting.compose import ForecastX

arimax_plus_exo = ForecastX(
    forecaster_y=StatsModelsARIMA(order=(1, 1, 0)),
    forecaster_X=ExponentialSmoothing(trend="add"),
)

arimax_plus_exo.fit(y_lg, X=X_lg, fh=fh_lg)  # no future X needed
y_pred_lg = arimax_plus_exo.predict()

plot_series(y_lg, y_pred_lg, labels=["History", "ARIMA + forecasted X"]);
""")

md("""
It works without us providing any future `X`. Let's look under the hood: the inner forecaster for `X` made its own forecasts, one column per exogenous variable, and these were fed into the ARIMA.
""")

code("""
arimax_plus_exo.forecaster_X_.predict()
""")

notes("""
**Keep this as the "shock moment"**: let the error appear, ask the room why. Then reveal `ForecastX`: the forecast of `X` is itself a forecast, so errors in `X` propagate into `y`. Only use exogenous variables whose future you can forecast well or know (calendar, planned prices, weather forecasts).

Keep it to ~8 min, no exercise.
""")

timing("""
**1:28-1:46 · 1.6 reduction + random forest exercise (18 min).** Two animations to explain, exercise 5 min. ✅ Checkpoint 1:46 (1:36 without 1.5): Block 1 done. Then **break 15 min** (1:46-2:01).
""")

md("""
## 1.6 Machine learning models as forecasters (*reduction*)

Can we use any scikit-learn regressor, like linear regression or random forest? Yes, with a trick called **reduction**: cut the series into windows, where the previous 12 values are the features and the next value is the target. Then it's a normal tabular regression problem.

<img src="imgs/reduction.gif" style="max-width: 800px" />

**But how do we forecast several steps ahead?** The model only predicts *one* step. `make_reduction` uses the **recursive** strategy by default: predict one step, treat the prediction as if it were an observation, and predict the next step from the updated window.

<img src="imgs/recursive.gif" style="max-width: 800px" />

`sktime` does the windowing and the recursion for us with `make_reduction`. The resulting object is a normal `sktime` forecaster.
""")

code("""
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import make_reduction

lr_forecaster = make_reduction(LinearRegression(), window_length=12)
lr_forecaster.fit(y, fh=fh)

plot_series(y, lr_forecaster.predict(), labels=["History", "Linear regression"]);
""")

exercise(
    "A non-linear model",
    """
Replace the linear regression with a `RandomForestRegressor` (or any other scikit-learn regressor), again with `window_length=12`.

Plot the forecast. **What do you notice at the end of the forecast? Why might this happen?**
""",
    '''
from sklearn.ensemble import RandomForestRegressor

rf_forecaster = ...
rf_forecaster.fit(...)

plot_series(...)
''',
    '''
from sklearn.ensemble import RandomForestRegressor

rf_forecaster = make_reduction(RandomForestRegressor(random_state=0), window_length=12)
rf_forecaster.fit(y, fh=fh)

plot_series(y, rf_forecaster.predict(), labels=["History", "Random forest"]);

# Answer: the forecast stays on the level of the past. A tree can only output values
# it has seen in training (averages of leaves), so it cannot extrapolate the trend.
''',
)

md("""
**Why is the forecast flat?** A tree predicts averages of training values. It can never output a value larger than the ones it has seen, so it cannot continue a trend.

<img src="imgs/trees.png" style="max-width: 700px" />
""")

notes("""
This is the **cliffhanger** into Block 2: the random forest forecast is flat/too low, it cannot extrapolate the upward trend. Let people guess before revealing. Backtest numbers later: RF ~8.6% vs ETS ~4.0%.

Good moment for the first 15-min break.
""")

# =====================================================================
# 2. PIPELINES
# =====================================================================
timing("""
**2:01-2:23 · 2.1 pipelines + exercise (22 min).** Exercise 7 min.
""")

md("""
---
# Block 2 · Pipelines, tuning, intervals (57 min)

## 2.1 Fixing the random forest with a pipeline

Trees cannot extrapolate a trend, but we can **remove the trend** before the model sees the data, and add it back afterwards. Also, the seasonal swings grow over time, which a `log` transform stabilises.

`sktime` has transformers for this. A `TransformedTargetForecaster` chains them with a forecaster, just like a scikit-learn `Pipeline`:

`log → detrend → random forest`, and the *inverse* of every step is applied automatically on `predict`.

<img src="imgs/pipeline.png" style="max-width: 1000px" />
""")

code("""
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.detrend import Detrender

pipe = TransformedTargetForecaster(
    [
        ("log", LogTransformer()),
        ("detrend", Detrender()),
        ("forecast", make_reduction(RandomForestRegressor(random_state=0), window_length=12)),
    ]
)

pipe.fit(y, fh=fh)
plot_series(y, pipe.predict(), labels=["History", "Log → detrend → random forest"]);
""")

md("""
Much better. And there is a shortcut: the `*` operator chains steps.
""")

code("""
from sktime.transformations.series.detrend import Deseasonalizer

rf = make_reduction(RandomForestRegressor(random_state=0), window_length=12)

pipe_long = LogTransformer() * Deseasonalizer(sp=12, model="multiplicative") * Detrender() * rf

pipe_long
""")

code("""
backtest(rf), backtest(pipe), backtest(pipe_long)
""")

notes("""
Expected MAPE: plain RF ~8.6% -> log+detrend+RF ~4-6% -> + deseasonalize ~4.2%. That is on par with ETS. Highlight the `*` operator and that pipelines are **forecasters themselves**: `fit`, `predict`, `evaluate` all work, so the next steps (tuning!) work on the whole pipeline.

Mention, but do not demo: `Imputer` for missing values, `pipe.save()` for persistence.
""")

exercise(
    "Build your own pipeline",
    """
Build **your own** pipeline with the `*` operator:

1. `Deseasonalizer(sp=12, model="multiplicative")`
2. `Detrender()`
3. `make_reduction(...)` with a regressor of your choice (e.g. `GradientBoostingRegressor`, `KNeighborsRegressor`, `Ridge`) and `window_length=12`

Backtest it with `backtest(...)`. Can you beat the pipelines above?
""",
    '''
from sklearn.ensemble import GradientBoostingRegressor  # or any other regressor

my_pipe = ...

backtest(my_pipe)
''',
    '''
from sklearn.ensemble import GradientBoostingRegressor

my_pipe = (
    Deseasonalizer(sp=12, model="multiplicative")
    * Detrender()
    * make_reduction(GradientBoostingRegressor(random_state=0), window_length=12)
)

backtest(my_pipe)
''',
)

timing("""
**2:23-2:40 · 2.2 tuning + exercise (17 min).** Exercise 6 min. Let the grid search run while you explain. If short on time: show the ETS tuning only and demo the pipeline tuning live.
""")

md("""
## 2.2 Hyperparameter tuning

Which parameters are good? Let `sktime` search, with backtesting as the judge: `ForecastingGridSearchCV` is a forecaster that wraps another forecaster and tunes it.

First on a simple model, ETS:
""")

code("""
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.split import ExpandingGreedySplitter

tuned_ets = ForecastingGridSearchCV(
    forecaster=ExponentialSmoothing(seasonal="multiplicative", sp=12),
    param_grid={
        "trend": ["add", "mul"],
        "damped_trend": [True, False],
    },
    cv=ExpandingGreedySplitter(test_size=12, folds=3),
    backend="threading",  # run the search in threads
)
tuned_ets.fit(y, fh=fh)

tuned_ets.best_params_
""")

code("""
tuned_ets.cv_results_.sort_values("rank_test_MeanAbsolutePercentageError")
""")

md("""
**Tuning a pipeline:** which parameters can we tune? `get_params()` lists them. The names follow the pattern `<step name>__<parameter>`.
""")

code("""
[p for p in pipe_long.get_params() if "window_length" in p or "max_depth" in p]
""")

exercise(
    "Tune the pipeline",
    """
Tune `pipe_long` with `ForecastingGridSearchCV`:

* `window_length` of the random forest: 6, 12, 24
* `max_depth` of the random forest: 5, 10, `None`

Use the parameter names from above. Print the best parameters.

*Careful: tuning on the same data you evaluate on gives optimistic numbers. For real projects, keep a final holdout.*
""",
    '''
tuned_pipe = ForecastingGridSearchCV(
    forecaster=pipe_long,
    param_grid={
        "...": [6, 12, 24],
        "...": [5, 10, None],
    },
    cv=ExpandingGreedySplitter(test_size=12, folds=3),
    backend="threading",  # run the search in threads
)
tuned_pipe.fit(y, fh=fh)

tuned_pipe.best_params_
''',
    '''
tuned_pipe = ForecastingGridSearchCV(
    forecaster=pipe_long,
    param_grid={
        "RecursiveTabularRegressionForecaster__window_length": [6, 12, 24],
        "RecursiveTabularRegressionForecaster__estimator__max_depth": [5, 10, None],
    },
    cv=ExpandingGreedySplitter(test_size=12, folds=3),
    backend="threading",  # run the search in threads
)
tuned_pipe.fit(y, fh=fh)

tuned_pipe.best_params_
''',
)

notes("""
The tuning here is deliberately small (9 combos x 3 folds). Not covered on purpose: randomized/Optuna search, AutoML, nested CV. Mention that they exist.
""")

timing("""
**2:40-2:55 · 2.3 prediction intervals (15 min).** Demo 10 min, exercise 5 min. ⏭️ The **exercise is SKIPPABLE (-5 min)**. Keep the demo, the wrap-up refers to it.
""")

md("""
## 2.3 How sure are we? Prediction intervals

A point forecast says "about 450 passengers". A **prediction interval** says how wrong that might be. A **90% interval** means: *we expect 90% of the future values to fall inside it.* Planning (stock, staff, budget) needs this range, not only the middle.

In `sktime` it is one more method: `predict_interval(coverage=0.9)`. Not every forecaster can do it natively. `ThetaForecaster` can:
""")

code("""
from sktime.forecasting.theta import ThetaForecaster

theta = ThetaForecaster(sp=12).fit(y_train, fh=fh)
theta_int = theta.predict_interval(coverage=0.9)

theta_int.head(3)
""")

code("""
plot_series(
    y_train.tail(48),
    y_test,
    theta.predict(),
    labels=["Train", "Test", "Theta + 90% interval"],
    pred_interval=theta_int,
);
""")

md("""
Looks reasonable, but **an interval is a claim, and we can check it**: how many of the true test values are really inside? That is the *empirical coverage*. For a 90% interval it should be close to 90%.
""")

code("""
from sktime.performance_metrics.forecasting.probabilistic import EmpiricalCoverage

EmpiricalCoverage(coverage=0.9)(y_test, theta_int)
""")

md("""
Far too low: the interval is **too narrow**, the model is overconfident. (Theta assumes the noise does not grow with the level, but our seasonal swings do.)

**Fix for any forecaster:** `ConformalIntervals` wraps a forecaster and turns its *past backtest errors* into prediction intervals. It works for models without native intervals, like our random forest pipeline. We put it as the last step, and `sktime` transforms the interval back to the original scale for us:
""")

code("""
from sktime.forecasting.conformal import ConformalIntervals

pipe_ci = (
    LogTransformer()
    * Deseasonalizer(sp=12, model="multiplicative")
    * Detrender()
    * ConformalIntervals(
        make_reduction(RandomForestRegressor(random_state=0), window_length=12),
        initial_window=48,
    )
)
pipe_ci.fit(y_train, fh=fh)
pipe_int = pipe_ci.predict_interval(coverage=0.9)

plot_series(
    y_train.tail(48),
    y_test,
    pipe_ci.predict(),
    labels=["Train", "Test", "RF pipeline + 90% interval"],
    pred_interval=pipe_int,
);
print("Coverage:", EmpiricalCoverage(coverage=0.9)(y_test, pipe_int))
""")

exercise(
    "Which coverage can you trust?",
    """
Compare the two models for the coverages 50%, 80% and 90%:

* `theta` (native intervals)
* `pipe_ci` (conformal intervals)

For each coverage, call `predict_interval(coverage=...)` and compute the empirical coverage on `y_test` with `EmpiricalCoverage`. Print a small table. **Which model keeps its promise?**

*Careful: the test set has only 24 points, so a few points more or less move the numbers a lot.*
""",
    '''
rows = {}
for c in [0.5, 0.8, 0.9]:
    cov = EmpiricalCoverage(coverage=c)
    rows[c] = {
        "theta": ...,
        "pipe_ci": ...,
    }

pd.DataFrame(rows).T
''',
    '''
rows = {}
for c in [0.5, 0.8, 0.9]:
    cov = EmpiricalCoverage(coverage=c)
    rows[c] = {
        "theta": cov(y_test, theta.predict_interval(coverage=c)),
        "pipe_ci": cov(y_test, pipe_ci.predict_interval(coverage=c)),
    }

pd.DataFrame(rows).T
''',
)

notes("""
**Do not teach the theory** (quantiles, distributions, pinball loss). Only the reading: "a 90% interval should contain 90% of the future values", and the check with empirical coverage. Message: intervals are model output like any other, so *backtest them too*. Native intervals depend on model assumptions (here: constant noise, wrong for multiplicative seasonality); conformal intervals only need backtest errors and work for every forecaster, including the pipeline and later the foundation model (4.4).

Numbers from the test run (90% coverage on the 24 test months): Theta ~29%, RF pipeline with conformal ~92%. There are more methods in `sktime`: `predict_quantiles`, `predict_var`, `predict_proba` (lecture 4/5 of the course). Mention they exist, ~20 min total for this section.
""")

timing("""
**2:55-2:58 · 2.4 leaderboard (3 min).** Just run the cell and read it out.
""")

md("""
## 2.4 Leaderboard so far

Let's collect all our models in one table. This is our reference for the end of the day.
""")

code("""
from sktime.forecasting.theta import ThetaForecaster

leaderboard = pd.Series(
    {
        "Seasonal naive": backtest(NaiveForecaster(strategy="last", sp=12)),
        "ETS": backtest(ExponentialSmoothing(trend="add", seasonal="mul", sp=12)),
        "Theta": backtest(ThetaForecaster(sp=12)),
        "Random forest (plain)": backtest(rf),
        "Pipeline: log → deseason → detrend → RF": backtest(pipe_long),
    },
    name="MAPE",
).sort_values()

leaderboard.to_frame().style.format("{:.1%}")
""")

# =====================================================================
# 3. PANEL AND HIERARCHICAL
# =====================================================================
timing("""
**2:58-3:05 · 3.1 panel and hierarchical data (7 min).** ✅ Checkpoint: Block 3 should start at 2:58 (2:43 if you skipped 1.5 and the 2.3 exercise).
""")

md("""
---
# Block 3 · Many series at once: panel and hierarchical data (32 min)

## 3.1 What is panel and hierarchical data?

In practice you rarely forecast one series. A retailer forecasts **thousands** of products and stores.

* **Panel data** = many series of the same kind (one per product, store, sensor, ...)
* **Hierarchical data** = panel data where the series **add up**: products → product groups → company total

<img src="imgs/hierarchical_data_tree.png" style="max-width: 600px" />

In `sktime` this is a `pandas` DataFrame with a **`MultiIndex`**: the last level is time, the levels before identify the series.
""")

code("""
from sktime.datasets import load_hierarchical_sales_toydata

y_h = load_hierarchical_sales_toydata()
y_h.head(8)
""")

md("""
Four series on the bottom level: two product lines with two product groups each. The `Aggregator` transformer adds the **aggregated series** (`__total`) of the hierarchy:
""")

code("""
from sktime.transformations.hierarchical.aggregate import Aggregator

y_h = Aggregator().fit_transform(y_h)

y_h.index.droplevel(-1).unique().to_frame(index=False)
""")

code("""
fig, ax = plt.subplots()
for (line, group), s in y_h.groupby(level=[0, 1]):
    s.droplevel([0, 1])["Sales"].plot(ax=ax, label=f"{line} / {group}", legend=True)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
ax.set_title("Sales per node of the hierarchy");
""")

notes("""
Use the picture to explain "coherent": the parent is the sum of its children. The `__total` naming is sktime's convention for aggregate nodes. Toy data: 5 years monthly, 7 series (4 bottom, 2 product lines, 1 total).
""")

timing("""
**3:05-3:15 · 3.2 broadcasting + exercise (10 min).** Exercise 5 min.
""")

md("""
## 3.2 Forecasting many series: it just works (*broadcasting*)

Every forecaster we saw before can be applied directly to panel data. `sktime` **broadcasts**: it fits one copy of the forecaster per series.

<img src="imgs/broadcasting.png" style="max-width: 800px" />
""")

code("""
y_h_train, y_h_test = temporal_train_test_split(y_h, test_size=12)
fh_h = np.arange(1, 13)

ets_h = ExponentialSmoothing(trend="add", seasonal="mul", sp=12)
ets_h.fit(y_h_train, fh=fh_h)
y_h_pred = ets_h.predict()

y_h_pred.head(4)
""")

code("""
total = ("__total", "__total")
plot_series(
    y_h_train.loc[total]["Sales"],
    y_h_test.loc[total]["Sales"],
    y_h_pred.loc[total]["Sales"],
    labels=["Train", "Test", "ETS forecast"],
    title="Company total",
);
""")

exercise(
    "A different forecaster for all series",
    """
Use a **different** forecaster for all seven series: `ThetaForecaster(sp=12)`. Then compute the MAE **per series** with `MeanAbsoluteError(multilevel="raw_values")`.

Which node of the hierarchy has the largest error? (Is that surprising?)
""",
    '''
from sktime.forecasting.theta import ThetaForecaster

theta_h = ...
theta_h.fit(...)
y_h_pred_theta = ...

mae_per_series = MeanAbsoluteError(multilevel="raw_values")
mae_per_series(...)
''',
    '''
from sktime.forecasting.theta import ThetaForecaster

theta_h = ThetaForecaster(sp=12)
theta_h.fit(y_h_train, fh=fh_h)
y_h_pred_theta = theta_h.predict()

mae_per_series = MeanAbsoluteError(multilevel="raw_values")
mae_per_series(y_h_test, y_h_pred_theta)

# The total has the largest absolute error simply because it has the largest values.
''',
)

timing("""
**3:15-3:30 · 3.3 incoherence and 3.4 reconciliation (15 min).** The 🌟 bonus exercise at the end of 3.4 is **SKIPPABLE** and not in the timeline (+5 min if you do it).
""")

md("""
## 3.3 The problem: forecasts that don't add up

We forecasted every series **independently**. So nothing forces the forecasts of the four products to add up to the forecast of the total. Let's check:
""")

code("""
def coherence_gap(y_pred):
    \"\"\"Largest gap between the sum of the 4 bottom series and the forecast of the total.\"\"\"
    idx = y_pred.index
    is_bottom = (idx.get_level_values(0) != "__total") & (idx.get_level_values(1) != "__total")
    bottom_sum = y_pred[is_bottom].groupby(level=-1).sum()["Sales"].to_numpy()
    total_fc = y_pred.loc[("__total", "__total")]["Sales"].to_numpy()
    return float(np.abs(bottom_sum - total_fc).max())


print(f"Largest gap between bottom sum and total: {coherence_gap(y_h_pred):.1f} units")
""")

md("""
Two departments would plan with different numbers. This is the **incoherence problem** of hierarchical forecasting.

<img src="imgs/hierarchical_data_tree_unreconciled.png" style="max-width: 600px" />
""")

md("""
## 3.4 The solution: reconciliation

**Reconciliation** adjusts the forecasts of all nodes, so that they add up again, *and* it tends to make them better, because information from all levels is combined.

<img src="imgs/hierarchical_data_tree_reconciled.png" style="max-width: 600px" />

In `sktime`, wrap your forecaster in a `ReconcilerForecaster`. We use the `mint_shrink` method (*MinT*: minimum-trace reconciliation, which takes the correlation of forecast errors into account):
""")

code("""
from sktime.forecasting.reconcile import ReconcilerForecaster

reconciled = ReconcilerForecaster(
    forecaster=ExponentialSmoothing(trend="add", seasonal="mul", sp=12),
    method="mint_shrink",
)
reconciled.fit(y_h_train)
y_h_pred_rec = reconciled.predict(fh=fh_h)

print(f"Gap before: {coherence_gap(y_h_pred):7.1f}")
print(f"Gap after:  {coherence_gap(y_h_pred_rec):7.1f}")
""")

md("""
This is the month with the largest gap, before and after reconciliation:

<img src="imgs/reconciliation.png" style="max-width: 800px" />
""")

code("""
mae_all = MeanAbsoluteError(multilevel="uniform_average")

pd.Series(
    {
        "independent ETS": mae_all(y_h_test, y_h_pred),
        "reconciled (MinT)": mae_all(y_h_test, y_h_pred_rec),
    },
    name="mean MAE over all 7 series",
).to_frame()
""")

md("""
On this tiny toy dataset the accuracy is about the same: **the main gain is coherence** (all numbers add up, so everyone plans with the same figures). On larger hierarchies, reconciliation often improves accuracy too.
""")

notes("""
Numbers from the test run: independent ETS 46.1, MinT 46.3, OLS 45.8, bottom-up 47.2 (mean MAE). Explain in two sentences what MinT does (weighted projection of the base forecasts onto the space of coherent forecasts, weights from the error covariance). Do not derive it.

Honest remark: reconciliation always gives coherent forecasts; whether the accuracy improves depends on the data. Point at lecture 3 of the forecasting course for the other methods.
""")

exercise(
    "Compare reconciliation methods",
    """
`sktime` has several reconciliation methods. Try `"bu"` (bottom-up: just sum the bottom forecasts), `"ols"` and `"wls_str"` in the same wrapper and compare the mean MAE as above. Does one stand out?
""",
    '''
scores = {}
for method in ["bu", "ols", "wls_str"]:
    rec = ReconcilerForecaster(
        forecaster=ExponentialSmoothing(trend="add", seasonal="mul", sp=12),
        method=...,
    )
    rec.fit(...)
    scores[method] = mae_all(...)

pd.Series(scores)
''',
    '''
scores = {}
for method in ["bu", "ols", "wls_str"]:
    rec = ReconcilerForecaster(
        forecaster=ExponentialSmoothing(trend="add", seasonal="mul", sp=12),
        method=method,
    )
    rec.fit(y_h_train)
    scores[method] = mae_all(y_h_test, rec.predict(fh=fh_h))

pd.Series(scores)
''',
    bonus=True,
)

# =====================================================================
# 4. FOUNDATION MODELS
# =====================================================================
timing("""
**3:30-3:40 · 4.1 zero-shot forecast and 4.2 backtest (10 min).** ✅ Checkpoint 3:30: Block 3 done.
""")

md("""
---
# Block 4 · Foundation models as drop-in forecasters (37 min)

## 4.1 What is a time series foundation model?

A **foundation model** (FM) is a model that someone else **pre-trained on a huge, diverse collection of time series**. You download the weights and forecast **without training on your data** (*zero-shot*): the model just reads your history as context.

<img src="imgs/global-fcst.png" style="max-width: 600px" />

`sktime` wraps many FMs as ordinary forecasters. You already know the interface:

<img src="imgs/fm-simple.png" style="max-width: 600px" />

We use **Chronos-Bolt (tiny)** from Amazon: 9 million parameters, runs on a laptop CPU, Apache-2.0 license.
""")

code("""
FALLBACK_FILE = "data/chronos_airline_fallback.csv"  # stored predictions, used only if the model cannot be loaded

y_train_fm = y.iloc[:-24]
fh_fm = np.arange(1, 25)

if CHRONOS_OK:
    chronos = ChronosForecaster("amazon/chronos-bolt-tiny")
    chronos.fit(y_train_fm)  # zero-shot: "fit" does not train anything on our data
    y_pred_chronos = chronos.predict(fh=fh_fm)
else:
    print("Using stored predictions (offline fallback).")
    y_pred_chronos = pd.read_csv(FALLBACK_FILE, index_col=0).squeeze("columns")
    y_pred_chronos.index = pd.PeriodIndex(y_pred_chronos.index, freq="M")

plot_series(y_train_fm.tail(48), y.iloc[-24:], y_pred_chronos, labels=["Train", "Truth", "Chronos"]);
""")

notes("""
Key sentence: **"fit" for a zero-shot model does not train anything**, it only stores the history as context for `predict`. Hence the speed.

The forecast looks quite good without any tuning or pipeline. Ask: "Is it better than our pipeline from Block 2?" -> that's the next cell. Trust, but verify.
""")

md("""
## 4.2 Trust, but verify: backtest on your own data

Marketing says that every foundation model is the best on some leaderboard. The only honest test is **your data and your backtest**. Same splitter, same metric as in Block 2:
""")

code("""
if CHRONOS_OK:
    leaderboard["Chronos-Bolt tiny (zero-shot)"] = backtest(
        ChronosForecaster("amazon/chronos-bolt-tiny")
    )

leaderboard.sort_values().to_frame().style.format("{:.1%}")
""")

notes("""
Expected MAPE: Chronos-Bolt tiny ~6.4%, i.e. clearly better than the plain random forest and seasonal naive, but **worse than ETS (4.0%) and the detrended pipeline (4.2%)**. Discuss: on a small, clean, strongly seasonal series, classical methods are hard to beat. FMs shine when you have many different series, little history for each, or no time to tune. And "tiny" is the smallest model.

Bigger Chronos variants (`chronos-bolt-small/base`) are a one-word change in the model name (download needed).
""")

timing("""
**3:40-3:50 · 4.3 context (10 min). ⏭️ SKIPPABLE (-10 min).** LLM-vs-TSFM table 3 min, context demo 4 min, discussion 3 min. The 🌟 bonus exercise is **SKIPPABLE** and not in the timeline.
""")

md("""
## 4.3 Context = the prompt

If you have used ChatGPT or any other large language model (LLM), you already know how a foundation model for time series works:

<img src="imgs/llm_vs_tsfm.png" style="max-width: 900px" />

| LLM | time series foundation model (TSFM) |
|---|---|
| **prompt** you write | **context**: the history you give the model |
| tokens (words, word pieces) | values of the series (Chronos-Bolt: patches of 16 values) |
| pre-trained on huge text corpora | pre-trained on huge collections of time series |
| *zero-shot*: just prompt it, no training | *zero-shot*: `fit` only stores the history as context |
| fine-tuning is optional | fine-tuning is optional |
| **continues** the text | **continues** the series = the forecast |
| maximal context window (in tokens) | maximal context window (Chronos-Bolt: 2048 values) |

Two differences: a TSFM has **no instructions**, its only job is "continue this series". And while the original Chronos generates its forecast step by step like an LLM generates words, Chronos-Bolt predicts the whole horizon in one go.

In `sktime`, the context is **whatever history you pass to the model**. Until now, the model saw *all* data before every backtest window (between 72 and 132 months). **What if we only show it the last 48 months?** Is there anything to learn from the series that is further back?
""")

code("""
fh_12 = np.arange(1, 13)


def backtest_chronos(context_length=None):
    \"\"\"MAPE per backtest window. The model only sees the last `context_length` months (None = all).\"\"\"
    errors = []
    for train_idx, test_idx in cv.split(y):
        if context_length is not None:
            train_idx = train_idx[-context_length:]  # cut the context
        chronos_ctx = ChronosForecaster("amazon/chronos-bolt-tiny").fit(y.iloc[train_idx])
        y_pred_ctx = chronos_ctx.predict(fh=fh_12)
        errors.append(mape(y.iloc[test_idx], y_pred_ctx))
    return pd.Series(errors, index=[f"window {i + 1}" for i in range(len(errors))])


if CHRONOS_OK:
    ctx_results = pd.DataFrame(
        {"all history": backtest_chronos(None), "last 48 months": backtest_chronos(48)}
    )
    ax = ctx_results.plot.bar(rot=0, color=[C_BLUE, C_ORANGE], title="Chronos-Bolt: MAPE per backtest window")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    print(ctx_results.mean().map("{:.1%}".format).to_string())
""")

md("""
Same model, same windows, same weights: only the **context** differs, and the error goes up substantially with the shorter context. The model has to infer *level, trend and seasonal pattern* from what it sees. With less history, it has less evidence for all three. (Not in every window though: in window 4 the shorter context was even better, which shows how much a single test window can mislead.)
""")

notes("""
Numbers from the test run (mean MAPE over the 6 windows): all history 6.4%, last 48 months 10.4%. Context sweep (months -> MAPE): 6 -> 15.1%, 12 -> 17.4%, 24 -> 13.2%, 36 -> 12.4%, 48 -> 10.4%, 60 -> 6.4%, 72 -> 6.2%, all -> 6.4%.

Careful with the story: nothing is trained, the only thing that changes is the input. Note the sweep is **not strictly monotone** (6 months beats 12 months) and there is a big jump between 48 and 60 months. We do not know exactly why, don't invent an explanation. One dataset is an anecdote, not proof of a general rule.

LLM parallel (2-3 min): prompt engineering for LLMs ~ context engineering for TSFMs: *what you put in the context matters*. TSFMs have no instruction following, no "chat". Original Chronos tokenizes values and samples autoregressively; Chronos-Bolt uses patches and predicts quantiles for the whole horizon directly. If someone asks "can I prompt it with text?": no, not these models.
""")

exercise(
    "How much context is enough?",
    """
Run the backtest for the context lengths 12, 24, 36, 48, 60 and 72 months with `backtest_chronos(...)`, and plot the **mean error against the context length**.

Is more context always better?
""",
    '''
contexts = [12, 24, 36, 48, 60, 72]

mean_error = {c: ... for c in contexts}

pd.Series(mean_error).plot(marker="o")
''',
    '''
contexts = [12, 24, 36, 48, 60, 72]

mean_error = {c: backtest_chronos(c).mean() for c in contexts}

ax = pd.Series(mean_error).plot(marker="o", color=C_ORANGE)
ax.set_xlabel("context length (months)")
ax.set_ylabel("mean MAPE")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
''',
    bonus=True,
)

md("""
**Is "more context is better" true?** It is a good rule of thumb, but it comes with caveats:

* ✅ **Usually helps, up to a point.** More history means more seasons, a better idea of the level and trend. In our run the error fell from about 17% (12 months) to about 6% (60 months).
* ➖ **Diminishing returns.** Beyond 60 months, nothing improved any more: 72 months and the full history were as good as 60.
* ⚠️ **Not always monotone.** More context is not a guarantee at every step: the curve is bumpy, as 6 months even beat 12 months.
* ⚠️ **Old data can be wrong data.** After a structural break (new product, pandemic, price change), old history describes a world that no longer exists. Then *less* context can win.
* 🚫 **Models have a maximum.** Chronos-Bolt uses at most the last 2048 values; older data is cut off. And a longer context costs more compute.

As with LLM prompts: **relevant** context beats **a lot of** context. And as always: test it on your own data.
""")

timing("""
**3:50-4:02 · 4.4 hierarchy, intervals, shampoo exercise (12 min).** Hierarchical demo 3 min, conformal intervals 3 min (⏭️ **SKIPPABLE, -3 min**), shampoo exercise 6 min.
""")

md("""
## 4.4 Everything you learned still works

Because the FM is a normal `sktime` forecaster, **hierarchical broadcasting** and **reconciliation** from Block 3 work out of the box. One copy of the model in memory, seven series.

(We use the `ols` reconciliation here: `mint_shrink` needs the model's *in-sample* errors, and a zero-shot foundation model cannot predict in-sample.)
""")

code("""
if CHRONOS_OK:
    chronos_h = ChronosForecaster("amazon/chronos-bolt-tiny")
    chronos_h.fit(y_h_train)
    y_h_pred_chronos = chronos_h.predict(fh=fh_h)

    chronos_rec = ReconcilerForecaster(
        ChronosForecaster("amazon/chronos-bolt-tiny"), method="ols"
    )
    chronos_rec.fit(y_h_train)
    y_h_pred_chronos_rec = chronos_rec.predict(fh=fh_h)

    print(f"Gap Chronos:            {coherence_gap(y_h_pred_chronos):7.1f}")
    print(f"Gap Chronos reconciled: {coherence_gap(y_h_pred_chronos_rec):7.1f}")
    display(
        pd.Series(
            {
                "ETS": mae_all(y_h_test, y_h_pred),
                "ETS reconciled": mae_all(y_h_test, y_h_pred_rec),
                "Chronos": mae_all(y_h_test, y_h_pred_chronos),
                "Chronos reconciled (OLS)": mae_all(y_h_test, y_h_pred_chronos_rec),
            },
            name="mean MAE",
        ).to_frame()
    )
""")

md("""
And **prediction intervals** work, too. Chronos itself returns point forecasts here, but you know the trick from 2.3: wrap *any* forecaster in `ConformalIntervals`.
""")

code("""
if CHRONOS_OK:
    from sktime.forecasting.conformal import ConformalIntervals

    chronos_conf = ConformalIntervals(
        ChronosForecaster("amazon/chronos-bolt-tiny"), initial_window=72
    )
    chronos_conf.fit(y_train_fm, fh=fh_fm)
    intervals = chronos_conf.predict_interval(coverage=0.9)

    plot_series(
        y.iloc[-72:],
        chronos_conf.predict(),
        labels=["History + truth", "Chronos + 90% interval"],
        pred_interval=intervals,
    );
""")

exercise(
    "Your turn: another dataset",
    """
Does Chronos beat simple models on a **different** series? Use the *shampoo sales* data (`load_shampoo_sales`, 36 months, strong trend).

Backtest `NaiveForecaster(strategy="last")`, `ExponentialSmoothing(trend="add")` and `ChronosForecaster("amazon/chronos-bolt-tiny")` with the small splitter given below, and print the MAE of each (use `MeanAbsoluteError`). Who wins?
""",
    '''
from sktime.datasets import load_shampoo_sales

y_shampoo = load_shampoo_sales()
cv_shampoo = ExpandingWindowSplitter(fh=[1, 2, 3], initial_window=24, step_length=3)

models = {
    "naive": ...,
    "ets": ...,
    "chronos": ...,
}

for name, model in models.items():
    res = evaluate(...)
    print(name, ...)
''',
    '''
from sktime.datasets import load_shampoo_sales

y_shampoo = load_shampoo_sales()
cv_shampoo = ExpandingWindowSplitter(fh=[1, 2, 3], initial_window=24, step_length=3)

models = {
    "naive": NaiveForecaster(strategy="last"),
    "ets": ExponentialSmoothing(trend="add"),
    "chronos": ChronosForecaster("amazon/chronos-bolt-tiny"),
}

for name, model in models.items():
    res = evaluate(model, cv_shampoo, y_shampoo, scoring=MeanAbsoluteError())
    print(f"{name:8} MAE: {res['test_MeanAbsoluteError'].mean():.1f}")
''',
)

notes("""
Numbers from the test run (MAE): naive 101, ETS 73, Chronos 112, i.e. Chronos is the *worst* here. Good teaching moment: a foundation model is no magic, on 36 noisy points with a trend it cannot beat ETS. The point is the *workflow*: swap one line, rerun, compare. Note that with 36 points and 4 windows, every conclusion is shaky. Bonus: change the model name to `amazon/chronos-bolt-small`.
""")

timing("""
**4:02-4:07 · 4.5 openness (5 min).** Talk only.
""")

md("""
## 4.5 Foundation models and openness

Before you put a foundation model into production, check **more than accuracy**:

* **Openness varies.** *Open weights* (like Chronos-Bolt on Hugging Face) can run **locally**: your data never leaves your machine. *Gated* models are paid APIs: you send your data to the provider.
* **Licenses are per artifact.** Weights, code, and data can have different licenses, and licenses can change. Read the model card.
* **Data protection and IP.** Sending business data to an API is a legal and strategic decision (GDPR, EU AI Act, trade secrets).
* **Reproducibility.** Pin library versions and weights. `sktime` ships stable wrappers with unified interfaces.

<img src="imgs/fm-license.png" style="max-width: 600px" />

**Rule of thumb:** *trust, but verify.* Benchmark on your own data and compare the pros and cons for your use case: simplicity, cost, interpretability, local vs. cloud.
""")

notes("""
This is the bridge to the next two agenda items of the meetup (open foundation models, hosting). Keep it to ~5 min; the other speaker will go deeper. Be careful with legal statements: you give rules of thumb, not legal advice.
""")

# =====================================================================
# WRAP-UP
# =====================================================================
timing("""
**4:07-4:12 · Wrap-up (5 min).** The full path ends at **4:12**. To end at 4:00, skip 1.5 and the 2.3 exercise (-15 min, ends 3:57), and 4.3 as well if you need more buffer (-10 min).
""")

md("""
---
# Wrap-up (5 min)

What you did today:

1. **`fit` / `predict` / `evaluate`**: one interface for naive, statistical, ML models, and foundation models
2. **Backtesting** instead of a single test set
3. **Pipelines** (`*` operator) and **tuning** on top of any forecaster
4. **Prediction intervals**: how to read them, and how to check them
5. **Panel and hierarchical data**: broadcasting and reconciliation
6. **Foundation models** are just another forecaster: *trust, but verify*

### Your recipe for Monday

1. **Start with a baseline**: a seasonal naive forecaster. Everything else has to beat it.
2. **Backtest**, don't trust a single test set: one splitter, one metric, for all models.
3. **Add structure** with pipelines: transform the target, add features, then fit the model.
4. **Tune** with backtesting as the judge, and keep a final holdout.
5. **Check the intervals**: does a 90% interval really cover 90%?
6. **Compare with a foundation model**: same interface, same backtest. Trust, but verify.

**Not covered today (but in `sktime`):** probabilistic forecasting in depth (quantiles, distributions), global forecasting, deep learning, anomaly detection, classification and clustering of time series, AutoML, benchmarking...

### Where to go next
* Docs and tutorials: <https://www.sktime.net>
* The full course this workshop is based on: <https://github.com/sktime/forecasting-course>
* Source code, issues, contributions: <https://github.com/sktime/sktime>
* The sktime community is friendly. Questions and first contributions are welcome.

**Thank you! 🙏**
""")

notes("""
Collect feedback, point to the course repo (this workshop is a condensed version of lectures 1, 2, 3 and 9 of it) and the next speaker.
""")


# =====================================================================
# BUILD
# =====================================================================
def build(participant: bool) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    cells = []
    for kind, src in CELLS:
        if kind in ("md", "task"):
            cells.append(nbf.v4.new_markdown_cell(src))
        elif kind == "code":
            cells.append(nbf.v4.new_code_cell(src))
        elif kind == "time" and not participant:
            c = nbf.v4.new_markdown_cell(src)
            c.metadata["tags"] = ["instructor"]
            cells.append(c)
        elif kind == "notes":
            continue  # instructor notes are switched off; the notes() calls stay in this file
        elif kind == "stub" and participant:
            cells.append(nbf.v4.new_code_cell(src))
        elif kind == "solution" and not participant:
            c = nbf.v4.new_code_cell(src)
            c.metadata["tags"] = ["solution"]
            cells.append(c)
    nb.cells = cells
    return nb


if __name__ == "__main__":
    for participant, name in [(False, "sktime_workshop_instructor.ipynb"), (True, "sktime_workshop.ipynb")]:
        nbf.write(build(participant), ROOT / name)
        print("wrote", name)
