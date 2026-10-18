import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import analysis
from model import train

OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)

BG = "#0a0d0b"
PANEL = "#111612"
TEXT = "#ecefe6"
MUTED = "#8d978a"
LIME = "#d4ff3a"
EMBER = "#ff7a45"
GRID = "#1d241e"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.grid": True,
        "grid.color": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "font.family": "DejaVu Sans",
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "savefig.facecolor": BG,
        "savefig.dpi": 160,
    }
)

df, lr, x_train, x_test, y_train, y_test = train()
m = float(lr.coef_[0])
b = float(lr.intercept_)
xs = np.linspace(df["cgpa"].min() - 0.2, df["cgpa"].max() + 0.2, 100)
line = m * xs + b


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name)
    plt.close()


fig, ax = plt.subplots(figsize=(11, 3.4))
ax.axis("off")
ax.grid(False)
ax.text(0.02, 0.62, "CGPA → PACKAGE", fontsize=36, fontweight="bold", color=TEXT, transform=ax.transAxes)
ax.text(
    0.02,
    0.22,
    "simple linear regression  ·  200 records  ·  R² 0.7807",
    fontsize=13,
    color=LIME,
    family="DejaVu Sans Mono",
    transform=ax.transAxes,
)
inset = fig.add_axes([0.72, 0.14, 0.25, 0.72])
inset.scatter(df["cgpa"], df["package"], s=9, color=MUTED, alpha=0.7)
inset.plot(xs, line, color=LIME, lw=3)
inset.set_xticks([])
inset.set_yticks([])
inset.grid(False)
for s in inset.spines.values():
    s.set_visible(False)
plt.savefig(OUT / "banner.png")
plt.close()

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.scatter(df["cgpa"], df["package"], s=34, color=LIME, alpha=0.75, edgecolor=BG, linewidth=0.6)
ax.set_xlabel("CGPA")
ax.set_ylabel("Package (LPA)")
ax.set_title("Raw data: 200 placements")
save("scatter.png")

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.scatter(x_train["cgpa"], y_train, s=30, color=MUTED, alpha=0.7, label=f"train ({len(x_train)})")
ax.scatter(x_test["cgpa"], y_test, s=40, color=EMBER, label=f"test ({len(x_test)})")
ax.plot(xs, line, color=LIME, lw=3, label=f"y = {m:.4f}x − {abs(b):.4f}")
ax.set_xlabel("CGPA")
ax.set_ylabel("Package (LPA)")
ax.set_title("The fitted regression line")
ax.legend()
save("fit.png")

fig, ax = plt.subplots(figsize=(9, 5.2))
sample = df.sample(n=6, random_state=4)
ax.scatter(df["cgpa"], df["package"], s=22, color=MUTED, alpha=0.35)
ax.plot(xs, line, color=LIME, lw=3)
for cg, pk in zip(sample["cgpa"], sample["package"]):
    pred = m * cg + b
    ax.plot([cg, cg], [pk, pred], color=EMBER, lw=2)
    ax.scatter([cg], [pk], s=60, color=EMBER, zorder=3)
    ax.scatter([cg], [pred], s=36, color=LIME, zorder=3)
ax.set_xlabel("CGPA")
ax.set_ylabel("Package (LPA)")
ax.set_title("Residuals: the vertical gaps the algorithm minimises")
save("residual_lines.png")

pred_test = lr.predict(x_test)
res = y_test.values - pred_test
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].scatter(x_test["cgpa"], res, s=40, color=EMBER)
axes[0].axhline(0, color=LIME, lw=2)
axes[0].set_xlabel("CGPA")
axes[0].set_ylabel("Residual (LPA)")
axes[0].set_title("Residuals vs CGPA (test set)")
axes[1].hist(res, bins=12, color=LIME, edgecolor=BG)
axes[1].axvline(0, color=EMBER, lw=2)
axes[1].set_xlabel("Residual (LPA)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual distribution")
save("residuals.png")

fig, ax = plt.subplots(figsize=(6.4, 6.4))
lo = min(y_test.min(), pred_test.min()) - 0.1
hi = max(y_test.max(), pred_test.max()) + 0.1
ax.plot([lo, hi], [lo, hi], color=LIME, lw=2, label="perfect prediction")
ax.scatter(y_test, pred_test, s=44, color=EMBER)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Actual package (LPA)")
ax.set_ylabel("Predicted package (LPA)")
ax.set_title("Actual vs predicted (test set)")
ax.legend(loc="upper left")
save("actual_vs_predicted.png")

X = x_train["cgpa"].values
Y = y_train.values
slopes = np.linspace(-0.2, 1.3, 300)
sse = [np.sum((Y - (s * X + (Y.mean() - s * X.mean()))) ** 2) for s in slopes]
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(slopes, sse, color=MUTED, lw=3)
best = np.sum((Y - lr.predict(x_train)) ** 2)
ax.scatter([m], [best], s=140, color=LIME, zorder=3)
ax.annotate(
    f"minimum at m = {m:.4f}",
    xy=(m, best),
    xytext=(m + 0.18, best + 55),
    color=LIME,
    fontsize=12,
    arrowprops=dict(arrowstyle="->", color=LIME),
)
ax.set_xlabel("slope m")
ax.set_ylabel("Sum of squared errors")
ax.set_title("Least squares: one slope has the lowest error")
save("sse_curve.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
axes[0].hist(df["cgpa"], bins=20, color=LIME, edgecolor=BG)
axes[0].set_title("CGPA distribution")
axes[0].set_xlabel("CGPA")
axes[0].set_ylabel("Students")
axes[1].hist(df["package"], bins=20, color=EMBER, edgecolor=BG)
axes[1].set_title("Package distribution")
axes[1].set_xlabel("Package (LPA)")
axes[1].set_ylabel("Students")
save("distributions.png")

fig, ax = plt.subplots(figsize=(9, 4.4))
pts = [4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5]
vals = [m * p + b for p in pts]
bars = ax.bar([str(p) for p in pts], vals, color=LIME, edgecolor=BG)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", color=TEXT, fontsize=10)
ax.set_xlabel("CGPA")
ax.set_ylabel("Predicted package (LPA)")
ax.set_title("What the model predicts across the CGPA range")
ax.set_ylim(0, max(vals) + 0.6)
save("prediction_ladder.png")


from scipy import stats

ana = analysis.build(df, lr, x_train, x_test, y_train, y_test)
T = ana["test"]
TR = ana["train"]
BASE = ana["base"]
GAIN = ana["gain"]
y_range = float(df["package"].max() - df["package"].min())

fig = plt.figure(figsize=(13, 6.4))
fig.suptitle("Regression metrics scoreboard · test set (40 students)", fontsize=17, fontweight="bold", color=TEXT, y=0.97)
cards = [
    ("MAE", f"{T['mae']:.4f}", "LPA · lower is better", f"{GAIN['mae']:.0f}% below baseline"),
    ("MSE", f"{T['mse']:.4f}", "LPA² · lower is better", f"{GAIN['mse']:.0f}% below baseline"),
    ("RMSE", f"{T['rmse']:.4f}", "LPA · lower is better", f"{GAIN['rmse']:.0f}% below baseline"),
    ("MAPE", f"{T['mape'] * 100:.2f}%", "lower is better", f"{GAIN['mape']:.0f}% below baseline"),
    ("R² SCORE", f"{T['r2']:.4f}", "higher is better", f"train {TR['r2']:.4f}"),
    ("ADJUSTED R²", f"{T['adj_r2']:.4f}", "higher is better", f"train {TR['adj_r2']:.4f}"),
    ("EXPLAINED VAR.", f"{T['evs']:.4f}", "higher is better", f"train {TR['evs']:.4f}"),
]
positions = [(0.02, 0.5), (0.27, 0.5), (0.52, 0.5), (0.77, 0.5), (0.02, 0.05), (0.27, 0.05), (0.52, 0.05)]
for (name, value, sub, foot), (px, py) in zip(cards, positions):
    ax = fig.add_axes([px, py, 0.22, 0.38])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.text(0.08, 0.84, name, color=EMBER, fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.08, 0.46, value, color=LIME, fontsize=28, fontweight="bold", transform=ax.transAxes)
    ax.text(0.08, 0.25, sub, color=MUTED, fontsize=9, transform=ax.transAxes)
    ax.text(0.08, 0.09, foot, color=TEXT, fontsize=9.5, transform=ax.transAxes)
ax = fig.add_axes([0.77, 0.05, 0.22, 0.38])
ax.axis("off")
ax.grid(False)
ax.text(0.02, 0.78, "baseline =", color=MUTED, fontsize=10, transform=ax.transAxes)
ax.text(0.02, 0.6, "always predict the", color=MUTED, fontsize=10, transform=ax.transAxes)
ax.text(0.02, 0.42, "training average", color=MUTED, fontsize=10, transform=ax.transAxes)
ax.text(0.02, 0.15, "MAE 0.6427 · RMSE 0.7448", color=TEXT, fontsize=9.5, transform=ax.transAxes)
plt.savefig(OUT / "metrics_scoreboard.png")
plt.close()

err = np.abs(ana["actual"] - ana["pred"])
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
srt = np.sort(err)
idx = np.arange(1, len(srt) + 1)
axes[0].vlines(idx, 0, srt, color=LIME, alpha=0.5, lw=2)
axes[0].scatter(idx, srt, color=LIME, s=40, zorder=3)
axes[0].axhline(T["mae"], color=EMBER, ls=":", lw=2, label=f"MAE {T['mae']:.3f}")
axes[0].axhline(T["rmse"], color=MUTED, ls=":", lw=2, label=f"RMSE {T['rmse']:.3f}")
axes[0].set_xlabel("Test students, sorted by size of miss")
axes[0].set_ylabel("Absolute error (LPA)")
axes[0].set_title("Every miss, smallest to largest")
axes[0].legend(loc="upper left")
counts, edges, patches = axes[1].hist(ana["ape"], bins=10, edgecolor=BG)
for patch, left in zip(patches, edges[:-1]):
    patch.set_facecolor(LIME if left + (edges[1] - edges[0]) / 2 <= T["mape"] * 100 else EMBER)
axes[1].axvline(T["mape"] * 100, color=TEXT, ls=":", lw=2, label=f"MAPE {T['mape'] * 100:.2f}%")
axes[1].set_xlabel("Absolute percentage error (%)")
axes[1].set_ylabel("Students")
axes[1].set_title("Percentage error distribution")
axes[1].legend()
save("error_metrics.png")

fig, ax = plt.subplots(figsize=(9, 4.6))
labels = ["MAE", "MSE", "RMSE", "MAPE"]
vals = [GAIN[k] for k in ("mae", "mse", "rmse", "mape")]
bars = ax.barh(labels[::-1], vals[::-1], color=LIME, edgecolor=BG)
for bar, v in zip(bars, vals[::-1]):
    ax.text(v - 2, bar.get_y() + bar.get_height() / 2, f"{v:.0f}% lower", ha="right", va="center", color=BG, fontsize=12, fontweight="bold")
ax.set_xlim(0, 100)
ax.set_xlabel("Improvement over predicting the average (%)")
ax.set_title("The model against a do-nothing baseline")
save("baseline_gain.png")

fig, ax = plt.subplots(figsize=(9, 5))
names = ["R²", "Adjusted R²", "Explained variance"]
keys = ["r2", "adj_r2", "evs"]
xpos = np.arange(3)
b1 = ax.bar(xpos - 0.2, [TR[k] for k in keys], 0.38, color=MUTED, label="train", edgecolor=BG)
b2 = ax.bar(xpos + 0.2, [T[k] for k in keys], 0.38, color=LIME, label="test", edgecolor=BG)
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{bar.get_height():.3f}", ha="center", color=TEXT, fontsize=10)
ax.set_xticks(xpos)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("R² family: train against test")
ax.legend(loc="lower right")
save("r2_family.png")

labels = ["R²", "Adj R²", "Explained\nvariance", "1 − MAPE", "1 − MAE\n/ range", "1 − RMSE\n/ range"]


def vec(m):
    return [max(m["r2"], 0), max(m["adj_r2"], 0), max(m["evs"], 0), 1 - m["mape"], 1 - m["mae"] / y_range, 1 - m["rmse"] / y_range]


angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
ax.set_facecolor(PANEL)
for name, m, col in (("train", TR, MUTED), ("test", T, LIME)):
    v = vec(m)
    v += v[:1]
    ax.plot(angles, v, color=col, lw=3, label=name)
    ax.fill(angles, v, color=col, alpha=0.18)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, color=TEXT, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color=MUTED, fontsize=8)
ax.grid(color=GRID)
ax.spines["polar"].set_color(GRID)
ax.set_title("Accuracy radar", pad=26, color=TEXT, fontweight="bold", fontsize=14)
ax.legend(loc="lower right", bbox_to_anchor=(1.2, -0.08))
plt.savefig(OUT / "accuracy_radar.png", bbox_inches="tight", pad_inches=0.4)
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
folds = [f"fold {i}" for i in range(1, 6)]
bars = ax.bar(folds, ana["cv"], color=LIME, edgecolor=BG, width=0.6)
for bar, v in zip(bars, ana["cv"]):
    ax.text(bar.get_x() + bar.get_width() / 2, v - 0.06, f"{v:.3f}", ha="center", color=BG, fontsize=11, fontweight="bold")
ax.axhline(ana["cv"].mean(), color=EMBER, ls=":", lw=2, label=f"mean {ana['cv'].mean():.3f} ± {ana['cv'].std():.3f}")
ax.set_ylim(0, 1.05)
ax.set_ylabel("R² score")
ax.set_title("5-fold cross-validation")
ax.legend(loc="lower right")
save("cv_scores.png")

fig, ax = plt.subplots(figsize=(9, 5))
used = [b for b in ana["bands"] if b["n"]]
bars = ax.bar([b["label"] for b in used], [b["mae"] for b in used], color=[LIME if b["mae"] < 0.3 else EMBER for b in used], edgecolor=BG, width=0.6)
for bar, b in zip(bars, used):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"MAPE {b['mape']:.1f}%\nn={b['n']}", ha="center", color=TEXT, fontsize=10)
ax.set_ylim(0, max(b["mae"] for b in used) * 1.35)
ax.set_xlabel("CGPA band (test students)")
ax.set_ylabel("MAE (LPA)")
ax.set_title("Where the model is strongest and weakest")
save("error_by_band.png")

fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
resid = ana["resid"]
pred = ana["pred"]
axes[0, 0].scatter(pred, resid, color=EMBER, s=45)
axes[0, 0].axhline(0, color=LIME, lw=2)
axes[0, 0].axhline(T["rmse"], color=MUTED, ls=":")
axes[0, 0].axhline(-T["rmse"], color=MUTED, ls=":")
axes[0, 0].set_xlabel("Fitted package (LPA)")
axes[0, 0].set_ylabel("Residual (LPA)")
axes[0, 0].set_title("Residuals vs fitted")
(osm, osr), (slope_q, icpt_q, r_q) = stats.probplot(resid, dist="norm")
axes[0, 1].scatter(osm, osr, color=LIME, s=45)
axes[0, 1].plot(osm, slope_q * osm + icpt_q, color=EMBER, ls="--", lw=2)
axes[0, 1].set_xlabel("Theoretical quantiles")
axes[0, 1].set_ylabel("Residual quantiles (LPA)")
axes[0, 1].set_title(f"Normal Q-Q plot (r² = {r_q ** 2:.3f})")
axes[1, 0].hist(resid, bins=10, color=LIME, alpha=0.7, edgecolor=BG, density=True)
xs_r = np.linspace(resid.min() - 0.2, resid.max() + 0.2, 200)
axes[1, 0].plot(xs_r, stats.norm.pdf(xs_r, resid.mean(), resid.std(ddof=1)), color=EMBER, lw=3)
axes[1, 0].set_xlabel("Residual (LPA)")
axes[1, 0].set_ylabel("Density")
axes[1, 0].set_title("Residual histogram with normal curve")
axes[1, 1].scatter(pred, np.sqrt(np.abs(resid / resid.std(ddof=1))), color=EMBER, s=45)
axes[1, 1].set_xlabel("Fitted package (LPA)")
axes[1, 1].set_ylabel("√|standardised residual|")
axes[1, 1].set_title("Scale-location (constant spread check)")
save("diagnostics.png")
