import numpy as np
import plotly.graph_objects as go

LIME = "#d4ff3a"
EMBER = "#ff7a45"
MUTED = "#8d978a"
TEXT = "#ecefe6"
GRID = "rgba(236,239,230,0.06)"
DIM = "rgba(236,239,230,0.2)"
MONO = "JetBrains Mono, monospace"


def _frame(fig, height, top=10, side=10):
    fig.update_layout(
        height=height,
        margin=dict(l=side, r=side, t=top, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO, color=MUTED, size=12),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#161d17",
            bordercolor=LIME,
            font=dict(family=MONO, color=TEXT),
        ),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def _marker():
    return dict(size=17, color=LIME, line=dict(color="#0a0d0b", width=3))


def package_distribution(df, shown):
    counts, edges = np.histogram(df["package"], bins=18)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [LIME if c <= shown else DIM for c in centers]
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            width=(edges[1] - edges[0]) * 0.9,
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{x:.2f} LPA<br>%{y} students<extra></extra>",
        )
    )
    fig.add_vline(
        x=shown,
        line=dict(color=EMBER, width=2, dash="dot"),
        annotation_text=f"you {shown:.2f}",
        annotation_font=dict(color=EMBER, family=MONO, size=12),
        annotation_position="top",
    )
    fig.update_xaxes(title="Package (LPA)")
    fig.update_yaxes(title="Students")
    return _frame(fig, 400, top=58)


def cgpa_distribution(df, cgpa):
    counts, edges = np.histogram(df["cgpa"], bins=18)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [LIME if c <= cgpa else DIM for c in centers]
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            width=(edges[1] - edges[0]) * 0.9,
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="CGPA %{x:.2f}<br>%{y} students<extra></extra>",
        )
    )
    fig.add_vline(
        x=cgpa,
        line=dict(color=EMBER, width=2, dash="dot"),
        annotation_text=f"you {cgpa:.2f}",
        annotation_font=dict(color=EMBER, family=MONO, size=12),
        annotation_position="top",
    )
    fig.update_xaxes(title="CGPA")
    fig.update_yaxes(title="Students")
    return _frame(fig, 400, top=58)


def fit_band(df, lr, cgpa, shown, rmse):
    xs = np.linspace(df["cgpa"].min() - 0.2, df["cgpa"].max() + 0.2, 60)
    line = lr.coef_[0] * xs + lr.intercept_
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["cgpa"],
            y=df["package"],
            mode="markers",
            marker=dict(size=8, color="rgba(236,239,230,0.35)"),
            hovertemplate="CGPA %{x:.2f}<br>%{y:.2f} LPA<extra></extra>",
        )
    )
    fig.add_trace(go.Scatter(x=xs, y=line + rmse, mode="lines", line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=line - rmse,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(212,255,58,0.12)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=xs, y=line, mode="lines", line=dict(color=LIME, width=3), hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=[cgpa],
            y=[shown],
            mode="markers",
            marker=_marker(),
            error_y=dict(type="data", array=[rmse], color=LIME, thickness=2, width=9),
            hovertemplate="You<br>CGPA %{x:.2f}<br>%{y:.2f} LPA<extra></extra>",
        )
    )
    fig.update_xaxes(title="CGPA", range=[df["cgpa"].min() - 0.4, df["cgpa"].max() + 0.4])
    fig.update_yaxes(title="Package (LPA)")
    return _frame(fig, 440)


def density(df, cgpa, shown):
    fig = go.Figure()
    fig.add_trace(
        go.Histogram2dContour(
            x=df["cgpa"],
            y=df["package"],
            ncontours=14,
            showscale=False,
            colorscale=[
                [0.0, "rgba(10,13,11,0)"],
                [0.15, "#18240f"],
                [0.5, "#5f8219"],
                [1.0, LIME],
            ],
            contours=dict(coloring="fill", showlines=False),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["cgpa"],
            y=df["package"],
            mode="markers",
            marker=dict(size=3.5, color="rgba(236,239,230,0.45)"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[cgpa],
            y=[shown],
            mode="markers",
            marker=_marker(),
            hovertemplate="You<br>CGPA %{x:.2f}<br>%{y:.2f} LPA<extra></extra>",
        )
    )
    fig.update_xaxes(title="CGPA")
    fig.update_yaxes(title="Package (LPA)")
    return _frame(fig, 400)


def ecdf(df, shown):
    values = np.sort(df["package"].values)
    pct = np.arange(1, len(values) + 1) / len(values) * 100
    at = float((df["package"] <= shown).mean() * 100)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=values,
            y=pct,
            mode="lines",
            line=dict(color=LIME, width=3, shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(212,255,58,0.10)",
            hovertemplate="%{x:.2f} LPA<br>%{y:.0f}% of students at or below<extra></extra>",
        )
    )
    fig.add_shape(type="line", x0=shown, x1=shown, y0=0, y1=at, line=dict(color=EMBER, width=2, dash="dot"))
    fig.add_shape(
        type="line",
        x0=values.min(),
        x1=shown,
        y0=at,
        y1=at,
        line=dict(color=EMBER, width=2, dash="dot"),
    )
    fig.add_trace(
        go.Scatter(
            x=[shown],
            y=[at],
            mode="markers+text",
            text=[f"{at:.0f}%"],
            textposition="top left",
            textfont=dict(color=EMBER, family=MONO, size=14),
            marker=_marker(),
            hovertemplate="You<br>%{x:.2f} LPA<extra></extra>",
        )
    )
    fig.update_xaxes(title="Package (LPA)")
    fig.update_yaxes(title="% of students at or below", range=[0, 105])
    return _frame(fig, 400)


def band_wheel(bands, band, shown):
    labels = [b["label"] for b in bands]
    avgs = [b["avg"] for b in bands]
    colors = [LIME if i == band else "rgba(236,239,230,0.22)" for i in range(len(bands))]
    fig = go.Figure()
    fig.add_trace(
        go.Barpolar(
            r=avgs,
            theta=labels,
            marker=dict(color=colors, line=dict(color="#0a0d0b", width=2)),
            hovertemplate="CGPA %{theta}<br>average %{r:.2f} LPA<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[shown],
            theta=[labels[band]],
            mode="markers",
            marker=dict(size=15, color=EMBER, line=dict(color="#0a0d0b", width=3)),
            hovertemplate="You<br>%{r:.2f} LPA<extra></extra>",
        )
    )
    top = max(max(avgs), shown) + 0.5
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, top], gridcolor=GRID, tickfont=dict(size=10), angle=90),
            angularaxis=dict(gridcolor=GRID, direction="clockwise", rotation=90, tickfont=dict(size=11, color=TEXT)),
        )
    )
    return _frame(fig, 420, top=40, side=50)


def _y_range(df):
    return float(df["package"].max() - df["package"].min())


def metric_vectors(m, y_range):
    return [
        max(m["r2"], 0.0),
        max(m["adj_r2"], 0.0),
        max(m["evs"], 0.0),
        1 - m["mape"],
        1 - m["mae"] / y_range,
        1 - m["rmse"] / y_range,
    ]


RADAR_LABELS = ["R²", "Adj R²", "Explained var.", "1 − MAPE", "1 − MAE/range", "1 − RMSE/range"]


def metric_radar(a, df):
    yr = _y_range(df)
    fig = go.Figure()
    for name, m, color, fill in (
        ("train", a["train"], MUTED, "rgba(141,151,138,0.12)"),
        ("test", a["test"], LIME, "rgba(212,255,58,0.2)"),
    ):
        v = metric_vectors(m, yr)
        fig.add_trace(
            go.Scatterpolar(
                r=v + v[:1],
                theta=RADAR_LABELS + RADAR_LABELS[:1],
                fill="toself",
                fillcolor=fill,
                line=dict(color=color, width=3),
                name=name,
                hovertemplate="%{theta}<br>%{r:.3f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)"),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 1], gridcolor=GRID, tickfont=dict(size=10), angle=90),
            angularaxis=dict(gridcolor=GRID, tickfont=dict(size=11, color=TEXT)),
        ),
    )
    return _frame(fig, 420, top=40, side=60)


def actual_vs_predicted(a):
    actual, pred = a["actual"], a["pred"]
    err = np.abs(actual - pred)
    lo = float(min(actual.min(), pred.min())) - 0.1
    hi = float(max(actual.max(), pred.max())) + 0.1
    rmse = a["test"]["rmse"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[lo, hi], y=[lo + rmse, hi + rmse], mode="lines", line=dict(width=0), hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo - rmse, hi - rmse],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(212,255,58,0.08)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=LIME, width=2, dash="dash"), hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=actual,
            y=pred,
            mode="markers",
            marker=dict(
                size=11,
                color=err,
                colorscale=[[0, LIME], [1, EMBER]],
                line=dict(color="#0a0d0b", width=1.5),
            ),
            hovertemplate="actual %{x:.2f}<br>predicted %{y:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Actual package (LPA)", range=[lo, hi])
    fig.update_yaxes(title="Predicted package (LPA)", range=[lo, hi])
    return _frame(fig, 420)


def residual_vs_fitted(a):
    rmse = a["test"]["rmse"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=a["pred"],
            y=a["resid"],
            mode="markers",
            marker=dict(size=10, color=EMBER, line=dict(color="#0a0d0b", width=1.5)),
            hovertemplate="fitted %{x:.2f}<br>residual %{y:+.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=LIME, width=2))
    for sign in (1, -1):
        fig.add_hline(y=sign * rmse, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_xaxes(title="Fitted package (LPA)")
    fig.update_yaxes(title="Residual (LPA)")
    return _frame(fig, 380)


def qq_plot(a):
    from scipy import stats

    (osm, osr), (slope, intercept, r) = stats.probplot(a["resid"], dist="norm")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            marker=dict(size=10, color=LIME, line=dict(color="#0a0d0b", width=1.5)),
            hovertemplate="theoretical %{x:.2f}<br>sample %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[osm.min(), osm.max()],
            y=[slope * osm.min() + intercept, slope * osm.max() + intercept],
            mode="lines",
            line=dict(color=EMBER, width=2, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(title="Theoretical normal quantiles")
    fig.update_yaxes(title="Residual quantiles (LPA)")
    return _frame(fig, 380)


def residual_hist(a):
    resid = a["resid"]
    counts, edges = np.histogram(resid, bins=10)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    xs = np.linspace(resid.min() - 0.2, resid.max() + 0.2, 120)
    mu, sd = resid.mean(), resid.std(ddof=1)
    pdf = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) * len(resid) * width
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=width * 0.92,
            marker=dict(color="rgba(212,255,58,0.65)", line=dict(width=0)),
            hovertemplate="%{x:+.2f} LPA<br>%{y} students<extra></extra>",
        )
    )
    fig.add_trace(go.Scatter(x=xs, y=pdf, mode="lines", line=dict(color=EMBER, width=3), hoverinfo="skip"))
    fig.update_xaxes(title="Residual (LPA)")
    fig.update_yaxes(title="Students")
    return _frame(fig, 380)


def abs_error_lollipop(a):
    err = np.sort(np.abs(a["actual"] - a["pred"]))
    idx = np.arange(1, len(err) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=idx,
            y=err,
            width=0.14,
            marker=dict(color="rgba(212,255,58,0.5)"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=err,
            mode="markers",
            marker=dict(size=10, color=LIME, line=dict(color="#0a0d0b", width=1.5)),
            hovertemplate="miss %{y:.2f} LPA<extra></extra>",
        )
    )
    fig.add_hline(
        y=a["test"]["mae"],
        line=dict(color=EMBER, width=2, dash="dot"),
        annotation_text=f"MAE {a['test']['mae']:.3f}",
        annotation_font=dict(color=EMBER, family=MONO, size=12),
        annotation_position="bottom left",
    )
    fig.add_hline(
        y=a["test"]["rmse"],
        line=dict(color=MUTED, width=2, dash="dot"),
        annotation_text=f"RMSE {a['test']['rmse']:.3f}",
        annotation_font=dict(color=MUTED, family=MONO, size=12),
        annotation_position="top left",
    )
    fig.update_xaxes(title="Test students, sorted by size of miss")
    fig.update_yaxes(title="Absolute error (LPA)")
    return _frame(fig, 380)


def mape_hist(a):
    counts, edges = np.histogram(a["ape"], bins=10)
    centers = (edges[:-1] + edges[1:]) / 2
    mape = a["test"]["mape"] * 100
    colors = [LIME if c <= mape else EMBER for c in centers]
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            width=(edges[1] - edges[0]) * 0.9,
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{x:.1f}% error<br>%{y} students<extra></extra>",
        )
    )
    fig.add_vline(
        x=mape,
        line=dict(color=TEXT, width=2, dash="dot"),
        annotation_text=f"MAPE {mape:.2f}%",
        annotation_font=dict(color=TEXT, family=MONO, size=12),
        annotation_position="top",
    )
    fig.update_xaxes(title="Absolute percentage error (%)")
    fig.update_yaxes(title="Students")
    return _frame(fig, 380, top=40)


def error_by_band(a):
    used = [b for b in a["bands"] if b["n"]]
    fig = go.Figure(
        go.Bar(
            x=[b["label"] for b in used],
            y=[b["mae"] for b in used],
            text=[f"{b['mape']:.1f}% · n={b['n']}" for b in used],
            textposition="outside",
            textfont=dict(color=TEXT, family=MONO, size=11),
            marker=dict(
                color=[b["mae"] for b in used],
                colorscale=[[0, LIME], [1, EMBER]],
                line=dict(width=0),
            ),
            hovertemplate="CGPA %{x}<br>MAE %{y:.3f} LPA<extra></extra>",
        )
    )
    top = max(b["mae"] for b in used) * 1.25
    fig.update_xaxes(title="CGPA band")
    fig.update_yaxes(title="MAE (LPA)", range=[0, top])
    return _frame(fig, 380, top=30)


def baseline_gain(a):
    labels = ["MAE", "MSE", "RMSE", "MAPE"]
    keys = ["mae", "mse", "rmse", "mape"]
    vals = [a["gain"][k] for k in keys]
    fig = go.Figure(
        go.Bar(
            y=labels[::-1],
            x=vals[::-1],
            orientation="h",
            text=[f"{v:.0f}% lower" for v in vals[::-1]],
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color="#0a0d0b", family=MONO, size=13),
            marker=dict(color=LIME, line=dict(width=0)),
            hovertemplate="%{y}: %{x:.1f}% lower than guessing the average<extra></extra>",
        )
    )
    fig.update_xaxes(title="Improvement over always guessing the average package (%)", range=[0, 100])
    fig.update_yaxes(title="")
    return _frame(fig, 380)


def cv_scores(a):
    folds = [f"fold {i}" for i in range(1, len(a["cv"]) + 1)]
    mean = float(a["cv"].mean())
    fig = go.Figure(
        go.Bar(
            x=folds,
            y=a["cv"],
            text=[f"{v:.3f}" for v in a["cv"]],
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color="#0a0d0b", family=MONO, size=12),
            marker=dict(color=LIME, line=dict(width=0)),
            width=0.55,
            hovertemplate="%{x}<br>R² %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=mean, line=dict(color=EMBER, width=2, dash="dot"))
    fig.add_annotation(
        x=1,
        xref="paper",
        xanchor="right",
        y=mean,
        yshift=16,
        text=f"mean {mean:.3f}",
        showarrow=False,
        font=dict(color=EMBER, family=MONO, size=12),
    )
    fig.update_yaxes(title="R² score", range=[0, 1.05])
    return _frame(fig, 380, top=20)


def train_test_bars(a):
    labels = ["R²", "Adj R²", "Explained var."]
    keys = ["r2", "adj_r2", "evs"]
    fig = go.Figure()
    for name, src, color in (("train", a["train"], MUTED), ("test", a["test"], LIME)):
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[src[k] for k in keys],
                name=name,
                text=[f"{src[k]:.3f}" for k in keys],
                textposition="outside",
                textfont=dict(color=TEXT, family=MONO, size=11),
                marker=dict(color=color, line=dict(width=0)),
                hovertemplate="%{x}<br>" + name + " %{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_yaxes(title="Score", range=[0, 1.05])
    return _frame(fig, 380, top=40)
