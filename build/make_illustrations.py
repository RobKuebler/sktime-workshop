"""Renders the workshop illustrations into imgs/.

    python build/make_illustrations.py

  - forecastx.gif:       what ForecastX does (forecast X first, then y)
  - reduction.gif:       sliding window -> regression table
  - recursive.gif:       recursive multi-step forecasting
  - backtest.gif:        expanding-window backtesting
  - split.png:           random vs temporal train/test split
  - trees.png:           trees cannot extrapolate a trend
  - pipeline.png:        what the random forest sees in a pipeline
  - broadcasting.png:    one forecaster, one copy per series
  - reconciliation.png:  incoherent vs reconciled forecasts
  - llm_vs_tsfm.png:     prompt <-> context, both models just continue the input

Uses the real Longley data and the real ForecastX forecasts from the notebook.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator

from sktime.datasets import load_longley
from sktime.forecasting.arima import StatsModelsARIMA
from sktime.forecasting.compose import ForecastX
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

warnings.filterwarnings("ignore")
IMGS = Path(__file__).resolve().parent.parent / "imgs"

C_Y, C_X, C_GREY, C_TEXT = "#1f6fb4", "#e07b00", "#8a8f98", "#222222"


def forecastx_animation():
    y, X = load_longley()
    x_col = "GNP"
    fh = np.arange(1, 5)

    fx = ForecastX(
        forecaster_y=StatsModelsARIMA(order=(1, 1, 0)),
        forecaster_X=ExponentialSmoothing(trend="add"),
    ).fit(y, X=X, fh=fh)
    y_fc = fx.predict()
    x_fc = fx.forecaster_X_.predict()[x_col]

    t_hist = y.index.year.to_numpy()
    t_fut = y_fc.index.year.to_numpy()
    # dashed segments start at the last observed point, so they connect to the solid line
    t_dash = np.r_[t_hist[-1], t_fut]
    y_dash = np.r_[y.iloc[-1], y_fc.to_numpy()]
    x_dash = np.r_[X[x_col].iloc[-1], x_fc.to_numpy()]

    fig, (ax_y, ax_x) = plt.subplots(
        2, 1, figsize=(8.5, 5.6), sharex=True, gridspec_kw={"hspace": 0.28}
    )
    fig.subplots_adjust(top=0.84, left=0.12, right=0.97, bottom=0.1)

    for ax, color, label in [
        (ax_y, C_Y, "y  (employment)"),
        (ax_x, C_X, f"X  ({x_col})"),
    ]:
        ax.axvspan(t_hist[-1] + 0.5, t_fut[-1] + 0.5, color="#f0f1f3", zorder=0)
        ax.set_ylabel(label, color=color, fontsize=11, fontweight="bold")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=9)
    ax_x.set_xlim(t_hist[0] - 0.5, t_fut[-1] + 0.5)
    ax_x.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax_y.plot(t_hist, y.to_numpy(), color=C_Y, lw=2.4)
    ax_x.plot(t_hist, X[x_col].to_numpy(), color=C_X, lw=2.4)
    ax_y.set_ylim(y.min() * 0.97, max(y.max(), y_fc.max()) * 1.04)
    ax_x.set_ylim(X[x_col].min() * 0.9, max(X[x_col].max(), x_fc.max()) * 1.06)

    ax_y.text(
        np.mean(t_fut), np.mean(ax_y.get_ylim()), "?",
        ha="center", va="center", fontsize=40, color=C_GREY, alpha=0.7,
    )
    q_mark = ax_y.texts[-1]
    ax_x.text(
        np.mean(t_fut), ax_x.get_ylim()[0] + 0.5 * np.diff(ax_x.get_ylim())[0], "future",
        ha="center", va="center", fontsize=10, color=C_GREY,
    )
    fut_label = ax_x.texts[-1]

    (line_x,) = ax_x.plot([], [], color=C_X, lw=2.4, ls="--", marker="o", ms=5)
    (line_y,) = ax_y.plot([], [], color=C_Y, lw=2.4, ls="--", marker="o", ms=5)

    arrows = []
    for tf, xf, yf in zip(t_fut, x_fc.to_numpy(), y_fc.to_numpy()):
        a = ConnectionPatch(
            xyA=(tf, xf), coordsA=ax_x.transData,
            xyB=(tf, ax_y.get_ylim()[0] + 0.12 * np.diff(ax_y.get_ylim())[0]), coordsB=ax_y.transData,
            arrowstyle="-|>", mutation_scale=18, lw=2.2, color=C_GREY, alpha=0,
        )
        fig.add_artist(a)
        arrows.append(a)

    caption = fig.text(0.5, 0.93, "", ha="center", va="center", fontsize=13, color=C_TEXT)
    sub = fig.text(0.5, 0.88, "", ha="center", va="center", fontsize=10.5, color=C_GREY)

    # timeline: (n_frames, caption, subcaption)
    n = len(fh)
    p0, p1, p2, p3, p4 = 10, 3 * n + 2, 2 * n + 2, 3 * n + 2, 14
    bounds = np.cumsum([p0, p1, p2, p3, p4])

    def update(f):
        if f < bounds[0]:
            caption.set_text("We want to forecast y for the next 4 years")
            sub.set_text("but the model also needs the future values of X ...")
        elif f < bounds[1]:
            k = min(n, (f - bounds[0]) // 3)
            caption.set_text("①  forecaster_X forecasts X")
            sub.set_text("e.g. ExponentialSmoothing, one forecast per X column")
            line_x.set_data(t_dash[: k + 1], x_dash[: k + 1]) if k else line_x.set_data([], [])
            fut_label.set_visible(False)
        elif f < bounds[2]:
            k = min(n, (f - bounds[1]) // 2 + 1)
            caption.set_text("②  forecasted X is fed into forecaster_y")
            sub.set_text("no future X needed from us")
            line_x.set_data(t_dash, x_dash)
            for i, a in enumerate(arrows):
                a.set_alpha(1.0 if i < k else 0.0)
        elif f < bounds[3]:
            k = min(n, (f - bounds[2]) // 3)
            caption.set_text("③  forecaster_y forecasts y")
            sub.set_text("e.g. ARIMA, using the forecasted X as input")
            q_mark.set_visible(False)
            line_y.set_data(t_dash[: k + 1], y_dash[: k + 1]) if k else line_y.set_data([], [])
        else:
            caption.set_text("ForecastX = a forecast of X  →  a forecast of y")
            sub.set_text("errors in the forecast of X propagate into y")
            line_y.set_data(t_dash, y_dash)
        if f == 0:  # reset for looping
            q_mark.set_visible(True)
            fut_label.set_visible(True)
            line_x.set_data([], [])
            line_y.set_data([], [])
            for a in arrows:
                a.set_alpha(0.0)
        return []

    anim = FuncAnimation(fig, update, frames=int(bounds[-1]), interval=100, blit=False)
    out = IMGS / "forecastx.gif"
    anim.save(out, writer=PillowWriter(fps=8), dpi=100)
    plt.close(fig)
    print("wrote", out, f"{out.stat().st_size / 1e3:.0f} KB")



# ---------------------------------------------------------------- helpers
C_OK = "#2e8b57"
PALETTE = ["#1f6fb4", "#e07b00", "#2e8b57", "#8e44ad", "#c0392b"]


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9)


def save_gif(fig, update, n_frames, name, durations):
    """durations: list of ms per frame (or single int)."""
    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)
    out = IMGS / name
    if isinstance(durations, int):
        durations = [durations] * n_frames
    # PillowWriter has no per-frame durations: render frames, assemble with PIL
    from io import BytesIO

    from PIL import Image

    frames = []
    for f in range(n_frames):
        update(f)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    plt.close(fig)
    print("wrote", out.name, f"{out.stat().st_size / 1e3:.0f} KB")


# ---------------------------------------------------------------- reduction
SERIES = [3, 5, 4, 6, 8, 7, 9, 11, 10, 12]
W = 4  # window length: 3 features + 1 target


def _series_axes(ax, n_steps, ymin=1, ymax=14):
    ax.set_xlim(0.3, n_steps + 0.7)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(1, n_steps + 1))
    ax.set_xlabel("time step", fontsize=10)
    ax.set_ylabel("value", fontsize=10)
    clean(ax)


def reduction_animation():
    n = len(SERIES)
    n_rows = n - W + 1
    steps = np.arange(1, n + 1)

    fig = plt.figure(figsize=(10, 5.0))
    ax = fig.add_axes([0.06, 0.14, 0.52, 0.62])
    tb = fig.add_axes([0.63, 0.08, 0.35, 0.72])
    tb.axis("off")
    tb.set_xlim(0, 4.3)
    tb.set_ylim(-(n_rows + 1.3), 0.7)
    _series_axes(ax, n)
    ax.plot(steps, SERIES, color=C_GREY, lw=1.2, zorder=1)
    ax.scatter(steps, SERIES, color="white", edgecolor=C_GREY, s=70, zorder=2)
    for t, v in zip(steps, SERIES):
        ax.text(t, v + 0.9, str(v), ha="center", fontsize=8, color=C_GREY)

    feat = ax.scatter([], [], color=C_Y, s=110, zorder=4)
    targ = ax.scatter([], [], color=C_X, s=110, zorder=4)
    from matplotlib.patches import Rectangle

    win = Rectangle((0, 1), 1, 13, facecolor="#dfe9f5", edgecolor=C_Y, lw=1.5, alpha=0.7, zorder=0)
    ax.add_patch(win)

    caption = fig.text(0.5, 0.95, "", ha="center", fontsize=13, color=C_TEXT)
    sub = fig.text(0.5, 0.895, "", ha="center", fontsize=10.5, color=C_GREY)

    # table header
    tb.text(2.0, 0.2, "regression dataset", ha="center", fontsize=11, fontweight="bold", color=C_TEXT)
    heads = ["x₁", "x₂", "x₃", "y"]
    for j, h in enumerate(heads):
        tb.text(j + 0.5 + (0.3 if j == 3 else 0), -0.8, h, ha="center", va="center", fontsize=12,
                fontweight="bold", color=C_X if j == 3 else C_Y)
    tb.plot([0, 4.3], [-1.3, -1.3], color=C_GREY, lw=1)
    tb.plot([3.15, 3.15], [-0.4, -(n_rows + 1.2)], color=C_GREY, lw=1, ls=":")
    row_artists = []
    for i in range(n_rows):
        yy = -(i + 1.9)
        hl = Rectangle((0, yy - 0.42), 4.3, 0.84, facecolor="#fff3cf", edgecolor="none", visible=False, zorder=0)
        tb.add_patch(hl)
        txt = []
        for j in range(W):
            v = SERIES[i + j]
            txt.append(tb.text(j + 0.5 + (0.3 if j == 3 else 0), yy, str(v), ha="center", va="center",
                               fontsize=12, color=C_X if j == 3 else C_Y, fontweight="bold" if j == 3 else None,
                               visible=False))
        row_artists.append((hl, txt))

    # timeline
    timeline = []  # (window_start (1-indexed, float), rows_visible, highlight_new, caption, sub)
    cap_a = f"A window of length {W} slides over the time series"
    sub_a = f"first {W - 1} values = features (x),  value {W} = target (y)"
    for i in range(n_rows):
        timeline += [(i + 1, i, False, cap_a, sub_a)] * 2
        timeline += [(i + 1, i + 1, True, cap_a, sub_a)] * 2
        if i < n_rows - 1:
            timeline += [(i + 1.5, i + 1, False, cap_a, sub_a)]
    end_cap = "A plain table: now any scikit-learn regressor can learn from it"
    timeline += [(n_rows, n_rows, False, end_cap, f"{n_rows} rows from {n} values. This is what make_reduction does.")] * 6

    def update(f):
        s, rows, hl_new, cap, sb = timeline[f]
        caption.set_text(cap)
        sub.set_text(sb)
        win.set_x(s - 0.4)
        win.set_width(W - 1 + 0.8)
        integer = abs(s - round(s)) < 1e-9
        if integer:
            k = int(round(s)) - 1
            feat.set_offsets(np.c_[steps[k : k + W - 1], SERIES[k : k + W - 1]])
            targ.set_offsets(np.c_[[steps[k + W - 1]], [SERIES[k + W - 1]]])
        else:
            feat.set_offsets(np.empty((0, 2)))
            targ.set_offsets(np.empty((0, 2)))
        for i, (hl, txt) in enumerate(row_artists):
            vis = i < rows
            for t in txt:
                t.set_visible(vis)
            hl.set_visible(vis and hl_new and i == rows - 1)
        return []

    durations = [450 if tl[3] == cap_a else 600 for tl in timeline]
    durations[-1] = 2500
    save_gif(fig, update, len(timeline), "reduction.gif", durations)


# ---------------------------------------------------------------- recursive
def recursive_animation():
    from sklearn.linear_model import LinearRegression

    n = len(SERIES)
    X = np.array([SERIES[i : i + W - 1] for i in range(n - W + 1)])
    yy = np.array([SERIES[i + W - 1] for i in range(n - W + 1)])
    model = LinearRegression().fit(X, yy)

    horizon = 3
    seq = list(SERIES)
    calls = []
    for _ in range(horizon):
        feats = seq[-(W - 1):]
        pred = float(model.predict(np.array([feats]))[0])
        calls.append((list(feats), pred))
        seq.append(pred)
    total = n + horizon

    fig = plt.figure(figsize=(10, 4.8))
    ax = fig.add_axes([0.06, 0.15, 0.56, 0.62])
    log = fig.add_axes([0.66, 0.15, 0.32, 0.62])
    log.axis("off")
    log.set_xlim(0, 1)
    log.set_ylim(0, 1)
    _series_axes(ax, total, ymin=1, ymax=16)
    steps = np.arange(1, total + 1)
    ax.axvspan(n + 0.5, total + 0.7, color="#f0f1f3", zorder=0)
    ax.plot(steps[:n], SERIES, color=C_GREY, lw=1.2, zorder=1)
    ax.scatter(steps[:n], SERIES, color="white", edgecolor=C_GREY, s=70, zorder=2)
    from matplotlib.patches import Rectangle

    win = Rectangle((0, 1), 1, 15, facecolor="#dfe9f5", edgecolor=C_Y, lw=1.5, alpha=0.7, zorder=0, visible=False)
    ax.add_patch(win)
    feat = ax.scatter([], [], color=C_Y, s=110, zorder=4)
    pred_sc = ax.scatter([], [], facecolor="white", edgecolor=C_X, s=110, lw=2.4, zorder=4)
    (pred_line,) = ax.plot([], [], color=C_X, lw=2, ls="--", zorder=3)
    caption = fig.text(0.5, 0.95, "", ha="center", fontsize=13, color=C_TEXT)
    sub = fig.text(0.5, 0.895, "", ha="center", fontsize=10.5, color=C_GREY)
    log.text(0, 0.98, "model calls", fontsize=11, fontweight="bold", color=C_TEXT, va="top")
    log_txt = [
        log.text(0, 0.84 - 0.2 * i, "", fontsize=10.5, family="monospace", color=C_TEXT, va="top")
        for i in range(horizon)
    ]

    timeline = []  # (stage, step_index)
    timeline += [(-1, 0)] * 3
    for h in range(horizon):
        timeline += [(0, h)] * 3  # window on last 3 values
        timeline += [(1, h)] * 3  # prediction appears
    timeline += [(2, horizon - 1)] * 6

    def update(f):
        stage, h = timeline[f]
        if stage == -1:
            caption.set_text("We trained the model on the table. Now: forecast 3 steps ahead")
            sub.set_text("but the model can only predict ONE step. What to do?")
            win.set_visible(False)
            feat.set_offsets(np.empty((0, 2)))
            pred_sc.set_offsets(np.empty((0, 2)))
            pred_line.set_data([], [])
            for t in log_txt:
                t.set_text("")
            return []
        caption.set_text("Recursive forecasting: predict one step, feed the prediction back in")
        k = n + h - (W - 1)  # 0-indexed start of the window of features
        done = h + (1 if stage >= 1 else 0)
        win.set_visible(True)
        win.set_x(k + 1 - 0.4)
        win.set_width(W - 2 + 0.8)
        feat.set_offsets(np.c_[steps[k : k + W - 1], seq[k : k + W - 1]])
        if stage == 0:
            sub.set_text(f"step {h + 1}: the last {W - 1} values are the features")
        elif stage == 1:
            sub.set_text(f"step {h + 1}: the model's prediction becomes a new data point")
        if stage == 2:
            sub.set_text("a multi-step forecast = repeated one-step forecasts")
        px = steps[n : n + done]
        py = seq[n : n + done]
        pred_sc.set_offsets(np.c_[px, py] if done else np.empty((0, 2)))
        pred_line.set_data(np.r_[steps[n - 1], px], np.r_[seq[n - 1], py])
        for i, t in enumerate(log_txt):
            if i < h or (i == h and stage >= 1) or stage == 2:
                feats, pr = calls[i]
                t.set_text(f"f({feats[0]:g}, {feats[1]:g}, {feats[2]:.4g}) = {pr:.1f}".replace(".0 ", " ")
                           if False else f"f({', '.join(f'{v:.4g}' for v in feats)})\n   = {pr:.1f}")
            elif i == h and stage == 0:
                feats, _ = calls[i]
                t.set_text(f"f({', '.join(f'{v:.4g}' for v in feats)})\n   = ?")
            else:
                t.set_text("")
        return []

    durations = [700] * len(timeline)
    durations[-1] = 3000
    save_gif(fig, update, len(timeline), "recursive.gif", durations)


# ---------------------------------------------------------------- backtest
def backtest_animation():
    from sktime.datasets import load_airline
    from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

    y = load_airline()
    vals = y.to_numpy()
    t = np.arange(len(y))
    mape = MeanAbsolutePercentageError()
    folds = []
    for cut in range(72, len(y) - 12 + 1, 12):
        f = ExponentialSmoothing(trend="add", seasonal="mul", sp=12).fit(y.iloc[:cut])
        pred = f.predict(fh=np.arange(1, 13)).to_numpy()
        folds.append((cut, pred, float(mape(y.iloc[cut : cut + 12], pred))))
    k_folds = len(folds)

    fig = plt.figure(figsize=(10, 5.4))
    ax = fig.add_axes([0.07, 0.34, 0.9, 0.5])
    bx = fig.add_axes([0.07, 0.07, 0.9, 0.17])
    clean(ax)
    clean(bx)
    ax.set_xlim(-2, len(y) + 1)
    ax.set_ylim(70, 680)
    ax.set_ylabel("passengers", fontsize=10)
    ax.set_xticks([])
    bx.set_xlim(0.4, k_folds + 0.6)
    bx.set_ylim(0, 0.085)
    bx.set_xticks(range(1, k_folds + 1))
    bx.set_xticklabels([f"window {i}" for i in range(1, k_folds + 1)], fontsize=8)
    bx.set_yticks([0, 0.04, 0.08])
    bx.set_yticklabels(["0%", "4%", "8%"], fontsize=8)
    bx.set_ylabel("MAPE", fontsize=9)
    ax.plot(t, vals, color="#d5d8dc", lw=1.2, zorder=1)
    (train_ln,) = ax.plot([], [], color=C_Y, lw=2.4, zorder=3)
    (truth_ln,) = ax.plot([], [], color=C_TEXT, lw=2, zorder=3)
    (pred_ln,) = ax.plot([], [], color=C_X, lw=2.4, ls="--", marker="o", ms=3.5, zorder=4)
    from matplotlib.patches import Rectangle

    shade = Rectangle((0, 0), 1, 1, color="#f0f1f3", zorder=0, visible=False)
    ax.add_patch(shade)
    caption = fig.text(0.5, 0.95, "", ha="center", fontsize=13, color=C_TEXT)
    sub = fig.text(0.5, 0.895, "", ha="center", fontsize=10.5, color=C_GREY)
    bars = bx.bar(range(1, k_folds + 1), [0] * k_folds, color=C_X, width=0.55)
    bar_txt = [bx.text(i + 1, 0, "", ha="center", va="bottom", fontsize=8.5, color=C_TEXT) for i in range(k_folds)]
    mean_txt = fig.text(0.97, 0.275, "", ha="right", fontsize=10.5, color=C_TEXT, fontweight="bold")

    timeline = []
    for i in range(k_folds):
        timeline += [(i, 0)] * 2 + [(i, 1)] * 3
    timeline += [(k_folds - 1, 2)] * 4

    def update(f):
        i, stage = timeline[f]
        cut, pred, err = folds[i]
        caption.set_text("Backtesting: repeat fit → predict → score on moving windows")
        train_ln.set_data(t[:cut], vals[:cut])
        shade.set_visible(True)
        shade.set_xy((cut - 0.5, 70))
        shade.set_width(12)
        shade.set_height(610)
        if stage == 0:
            sub.set_text(f"window {i + 1}: fit on the past (blue), hide the next 12 months")
            truth_ln.set_data([], [])
            pred_ln.set_data([], [])
        else:
            sub.set_text(f"window {i + 1}: forecast (orange) vs. truth (black)  →  MAPE {err:.1%}")
            truth_ln.set_data(t[cut - 1 : cut + 12], vals[cut - 1 : cut + 12])
            pred_ln.set_data(t[cut : cut + 12], pred)
        for j in range(k_folds):
            shown = j < i or (j == i and stage >= 1)
            h = folds[j][2] if shown else 0
            bars[j].set_height(h)
            bar_txt[j].set_position((j + 1, h))
            bar_txt[j].set_text(f"{h:.1%}" if shown else "")
        if stage == 2:
            sub.set_text("the average over all windows is a much more reliable score than one test set")
            mean_txt.set_text(f"mean MAPE: {np.mean([x[2] for x in folds]):.1%}")
        else:
            mean_txt.set_text("")
        return []

    durations = [600] * len(timeline)
    durations[-1] = 3000
    save_gif(fig, update, len(timeline), "backtest.gif", durations)


# ---------------------------------------------------------------- split
def split_png():
    from sktime.datasets import load_airline

    y = load_airline()
    vals = y.to_numpy()
    t = np.arange(len(y))
    rng = np.random.default_rng(1)
    test_idx = rng.choice(len(y), 24, replace=False)
    mask = np.zeros(len(y), bool)
    mask[test_idx] = True
    tmask = t >= len(y) - 24

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
    for ax, m, title, ok in [
        (axs[0], mask, "random split (like in scikit-learn)", False),
        (axs[1], tmask, "temporal split (what we need)", True),
    ]:
        ax.plot(t, vals, color="#d5d8dc", lw=1, zorder=1)
        ax.scatter(t[~m], vals[~m], s=14, color=C_Y, zorder=2, label="train")
        ax.scatter(t[m], vals[m], s=22, color=C_X, zorder=3, label="test")
        ax.set_title(title, fontsize=12, color=C_OK if ok else "#c0392b", fontweight="bold")
        ax.set_xticks([])
        clean(ax)
    axs[0].set_ylabel("passengers", fontsize=10)
    axs[0].text(0.03, 0.9, "test points sit between training points:\nthe model has already seen the future ✗",
                transform=axs[0].transAxes, fontsize=9.5, color="#c0392b", va="top")
    axs[1].text(0.03, 0.9, "the test set is the end of the series:\nwe only train on the past ✓",
                transform=axs[1].transAxes, fontsize=9.5, color=C_OK, va="top")
    axs[1].legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMGS / "split.png", dpi=110)
    plt.close(fig)
    print("wrote split.png")


# ---------------------------------------------------------------- trees
def trees_png():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    t_train = np.arange(0, 30)
    y_train = 2.0 * t_train + 10 + rng.normal(0, 4, len(t_train))
    t_all = np.arange(0, 45)
    rf = RandomForestRegressor(random_state=0).fit(t_train.reshape(-1, 1), y_train)
    lr = LinearRegression().fit(t_train.reshape(-1, 1), y_train)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axvspan(29.5, 45, color="#f0f1f3", zorder=0)
    ax.scatter(t_train, y_train, color=C_GREY, s=22, zorder=2, label="training data")
    ax.plot(t_all, lr.predict(t_all.reshape(-1, 1)), color=C_Y, lw=2.2, label="linear regression: continues the trend")
    ax.plot(t_all, rf.predict(t_all.reshape(-1, 1)), color=C_X, lw=2.2, label="random forest: stays flat")
    ax.text(37, 20, "future", ha="center", color=C_GREY, fontsize=11)
    ax.set_xlabel("time", fontsize=10)
    ax.set_ylabel("value", fontsize=10)
    ax.set_title("A tree can only predict values it has seen in training", fontsize=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    clean(ax)
    fig.tight_layout()
    fig.savefig(IMGS / "trees.png", dpi=110)
    plt.close(fig)
    print("wrote trees.png")


# ---------------------------------------------------------------- pipeline
def pipeline_png():
    from sktime.datasets import load_airline
    from sktime.transformations.series.boxcox import LogTransformer
    from sktime.transformations.series.detrend import Deseasonalizer, Detrender

    y = load_airline()
    y1 = LogTransformer().fit_transform(y)
    y2 = Deseasonalizer(sp=12, model="multiplicative").fit_transform(y1)
    y3 = Detrender().fit_transform(y2)
    panels = [
        (y, "original", None, C_GREY),
        (y1, "① LogTransformer\nstabilises the growing swings", None, C_Y),
        (y2, "② Deseasonalizer\nremoves the yearly pattern", None, C_Y),
        (y3, "③ Detrender\nremoves the trend", None, C_Y),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (s, title, step, col) in zip(axs, panels):
        ax.plot(np.arange(len(s)), s.to_numpy(), color=col, lw=1.6)
        ax.set_title(title, fontsize=11, color=C_TEXT)
        ax.set_xticks([])
        clean(ax)
    axs[3].set_facecolor("#fff3cf")
    axs[3].set_title("③ Detrender: removes the trend\n= what the forest sees", fontsize=11, color=C_X, fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.78, bottom=0.2, wspace=0.28)
    for i in range(3):
        a, b = axs[i].get_position(), axs[i + 1].get_position()
        xm = (a.x1 + b.x0) / 2
        fig.add_artist(plt.matplotlib.patches.FancyArrowPatch(
            (xm - 0.012, 0.5), (xm + 0.012, 0.5), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=16, color=C_GREY, lw=2))
    fig.text(0.5, 0.06,
             "fit: apply the steps from left to right, train the model on the last panel.    "
             "predict: forecast on the last panel, then invert every step from right to left.",
             ha="center", fontsize=10, color=C_TEXT)
    fig.savefig(IMGS / "pipeline.png", dpi=110)
    plt.close(fig)
    print("wrote pipeline.png")


# ---------------------------------------------------------------- hierarchy data
def _hier_forecasts():
    from sktime.datasets import load_hierarchical_sales_toydata
    from sktime.forecasting.reconcile import ReconcilerForecaster
    from sktime.split import temporal_train_test_split
    from sktime.transformations.hierarchical.aggregate import Aggregator

    y_h = Aggregator().fit_transform(load_hierarchical_sales_toydata())
    tr, te = temporal_train_test_split(y_h, test_size=12)
    fh = np.arange(1, 13)
    ets = ExponentialSmoothing(trend="add", seasonal="mul", sp=12)
    pred = ets.fit(tr, fh=fh).predict()
    rec = ReconcilerForecaster(
        ExponentialSmoothing(trend="add", seasonal="mul", sp=12), method="mint_shrink"
    ).fit(tr)
    pred_rec = rec.predict(fh=fh)
    return tr, te, pred, pred_rec


def broadcasting_png():
    tr, te, pred, _ = _hier_forecasts()
    keys = [
        (("Food preparation", "Hobs"), "Hobs"),
        (("Food preservation", "Freezers"), "Freezers"),
        (("__total", "__total"), "Total"),
    ]
    fig, axs = plt.subplots(3, 3, figsize=(11, 5.6), gridspec_kw={"width_ratios": [3, 1.3, 3], "wspace": 0.12})
    for r, (key, name) in enumerate(keys):
        s = tr.loc[key]["Sales"].to_numpy()
        p = pred.loc[key]["Sales"].to_numpy()
        a_in, a_mid, a_out = axs[r]
        a_in.plot(np.arange(len(s)), s, color=PALETTE[r], lw=1.6)
        a_in.set_ylabel(name, fontsize=10, color=PALETTE[r], fontweight="bold")
        a_out.plot(np.arange(len(s)), s, color="#d5d8dc", lw=1.2)
        a_out.plot(np.arange(len(s) - 1, len(s) + len(p)), np.r_[s[-1], p], color=PALETTE[r], lw=2, ls="--")
        for a in (a_in, a_out):
            a.set_xticks([])
            a.set_yticks([])
            clean(a)
        a_mid.axis("off")
        a_mid.set_xlim(0, 1)
        a_mid.set_ylim(0, 1)
        a_mid.annotate("", xy=(1.0, 0.5), xytext=(0.0, 0.5), arrowprops=dict(arrowstyle="-|>", color=C_GREY, lw=2))
        a_mid.text(0.5, 0.62, f"copy #{r + 1}", ha="center", fontsize=9.5, color=C_GREY)
        a_mid.text(0.5, 0.18, "ExponentialSmoothing\n(own parameters)", ha="center", fontsize=8, color=C_TEXT)
    axs[0][0].set_title("history of each series", fontsize=11)
    axs[0][2].set_title("forecast of each series", fontsize=11)
    axs[0][1].set_title("fit & predict", fontsize=11)
    fig.suptitle("Broadcasting: sktime fits one copy of the forecaster per series (here 3 of 7 shown)",
                 fontsize=12, y=0.99)
    fig.subplots_adjust(top=0.88, left=0.06, right=0.98, bottom=0.04)
    fig.savefig(IMGS / "broadcasting.png", dpi=110)
    plt.close(fig)
    print("wrote broadcasting.png")


def reconciliation_png():
    _, _, pred, pred_rec = _hier_forecasts()

    def parts(p):
        idx = p.index
        bottom = (idx.get_level_values(0) != "__total") & (idx.get_level_values(1) != "__total")
        b = p[bottom]["Sales"]
        tot = p.loc[("__total", "__total")]["Sales"]
        dates = tot.index
        bm = np.array([b.xs(d, level=-1).to_numpy() for d in dates])  # (months, 4)
        return dates, bm, tot.to_numpy()

    dates, bm, tot = parts(pred)
    gaps = np.abs(bm.sum(1) - tot)
    m = int(np.argmax(gaps))
    _, bm_r, tot_r = parts(pred_rec)
    names = ["Hobs", "Ovens", "Freezers", "Fridges"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, b, tt, title, ok in [
        (axs[0], bm[m], tot[m], "independent forecasts", False),
        (axs[1], bm_r[m], tot_r[m], "after reconciliation", True),
    ]:
        ax.bar(0, tt, color="#8a8f98", width=0.55)
        ax.text(0, tt + 6, f"{tt:.0f}", ha="center", fontsize=10)
        bottom = 0
        for i, v in enumerate(b):
            ax.bar(1, v, bottom=bottom, color=PALETTE[i], width=0.55, edgecolor="white")
            ax.text(1, bottom + v / 2, f"{names[i]}\n{v:.0f}", ha="center", va="center", fontsize=8, color="white")
            bottom += v
        ax.text(1, bottom + 6, f"{bottom:.0f}", ha="center", fontsize=10)
        gap = bottom - tt
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["forecast of\nthe total", "sum of the 4\nbottom forecasts"], fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", color=C_OK if ok else "#c0392b")
        ax.set_ylim(0, 700)
        ax.text(0.5, 0.97, ("gap: %.0f units ✗" % abs(gap)) if not ok else "gap: 0 ✓", transform=ax.transAxes,
                ha="center", va="top", fontsize=11.5, color=C_OK if ok else "#c0392b", fontweight="bold")
        clean(ax)
    axs[0].set_ylabel(f"forecast of sales, {dates[m]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(IMGS / "reconciliation.png", dpi=110)
    plt.close(fig)
    print("wrote reconciliation.png")



# ---------------------------------------------------------------- LLM vs TSFM
def llm_vs_tsfm_png():
    from matplotlib.patches import FancyBboxPatch

    from sktime.datasets import load_airline
    from sktime.forecasting.chronos import ChronosForecaster

    y = load_airline()
    context = y.iloc[:-12].iloc[-60:]
    pred = ChronosForecaster("amazon/chronos-bolt-tiny").fit(context).predict(fh=np.arange(1, 13))

    fig = plt.figure(figsize=(11.5, 5.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y_, w, h, text, fc, ec, ls="-", fs=11, color=C_TEXT, bold=False):
        ax.add_patch(FancyBboxPatch((x, y_), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=fc, edgecolor=ec, lw=1.8, ls=ls))
        ax.text(x + w / 2, y_ + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else None)

    def arrow(x0, x1, y_):
        ax.annotate("", xy=(x1, y_), xytext=(x0, y_), arrowprops=dict(arrowstyle="-|>", color=C_GREY, lw=2.2))

    # column headers
    for x, t in [(2.35, "input"), (5.75, "model"), (9.0, "output")]:
        ax.text(x, 5.35, t, ha="center", fontsize=12.5, fontweight="bold", color=C_GREY)

    # ---- row 1: LLM
    ax.text(0.05, 4.0, "LLM", fontsize=14, fontweight="bold", color=C_TEXT, va="center")
    ax.text(0.05, 3.62, "text", fontsize=10, color=C_GREY, va="center")
    for i, tok in enumerate(["Once", "upon", "a"]):
        box(1.0 + i * 1.15, 3.75, 1.05, 0.5, tok, "#dfe9f5", C_Y)
    ax.text(1.0 + 3 * 1.15 - 0.02, 4.45, "prompt", fontsize=10, color=C_Y, ha="right", fontweight="bold")
    arrow(4.55, 5.0, 4.0)
    box(5.0, 3.55, 1.5, 0.9, "pre-trained\nLLM", "#eceef1", C_GREY, bold=True)
    arrow(6.55, 7.0, 4.0)
    for i, tok in enumerate(["time", "there", "was"]):
        box(7.1 + i * 1.3, 3.75, 1.2, 0.5, tok, "white", C_X, ls="--")
    ax.text(7.1, 4.45, "continuation", fontsize=10, color=C_X, fontweight="bold")

    # ---- row 2: TSFM
    ax.plot([0.05, 11.4], [2.75, 2.75], color="#e3e5e8", lw=1)
    ax.text(0.05, 1.45, "TSFM", fontsize=14, fontweight="bold", color=C_TEXT, va="center")
    ax.text(0.05, 1.08, "numbers", fontsize=10, color=C_GREY, va="center")
    a1 = fig.add_axes([1.0 / 11.5, 0.9 / 5.6, 3.35 / 11.5, 1.35 / 5.6])
    a1.plot(np.arange(len(context)), context.to_numpy(), color=C_Y, lw=2)
    a1.set_xticks([])
    a1.set_yticks([])
    a1.set_ylim(y.min() * 0.9, y.max() * 1.02)
    clean(a1)
    ax.text(4.35, 2.45, "context", fontsize=10, color=C_Y, ha="right", fontweight="bold")
    arrow(4.55, 5.0, 1.55)
    box(5.0, 1.1, 1.5, 0.9, "pre-trained\nTSFM", "#eceef1", C_GREY, bold=True)
    arrow(6.55, 7.0, 1.55)
    a2 = fig.add_axes([7.1 / 11.5, 0.9 / 5.6, 3.9 / 11.5, 1.35 / 5.6])
    n = len(context)
    a2.plot(np.arange(n), context.to_numpy(), color="#c5d5ea", lw=1.6)
    a2.plot(np.arange(n - 1, n + 12), np.r_[context.iloc[-1], pred.to_numpy()], color=C_X, lw=2.2, ls="--")
    a2.set_xticks([])
    a2.set_yticks([])
    a2.set_ylim(y.min() * 0.9, y.max() * 1.02)
    clean(a2)
    ax.text(7.1, 2.45, "continuation", fontsize=10, color=C_X, fontweight="bold")

    # ---- parallels
    ax.text(5.75, 0.38,
            "prompt  ↔  context       ·       tokens  ↔  values (or patches of values)       ·       "
            "writing text  ↔  forecasting",
            ha="center", fontsize=10.5, color=C_TEXT)
    ax.text(5.75, 0.08, "same idea: a model pre-trained on huge amounts of data, used without any training on your side (zero-shot)",
            ha="center", fontsize=9.5, color=C_GREY)
    fig.savefig(IMGS / "llm_vs_tsfm.png", dpi=110)
    plt.close(fig)
    print("wrote llm_vs_tsfm.png")


if __name__ == "__main__":
    IMGS.mkdir(exist_ok=True)
    forecastx_animation()
    reduction_animation()
    recursive_animation()
    backtest_animation()
    split_png()
    trees_png()
    pipeline_png()
    broadcasting_png()
    reconciliation_png()
    llm_vs_tsfm_png()
