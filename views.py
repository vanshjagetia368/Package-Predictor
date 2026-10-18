import random
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import charts
from insights import build as build_insights
from model import predict

FRONTEND = Path(__file__).parent / "frontend" / "cgpa_input"
_cgpa_input = components.declare_component("cgpa_input", path=str(FRONTEND))


def go_back():
    st.session_state.page = "input"
    st.session_state.visit += 1


def dots():
    rnd = random.Random(11)
    items = "".join(
        f'<i style="--x:{rnd.uniform(0, 100):.1f}vw;--s:{rnd.choice([2, 3, 4, 6])}px;'
        f'--t:{rnd.uniform(14, 32):.1f}s;--d:-{rnd.uniform(0, 30):.1f}s"></i>'
        for _ in range(28)
    )
    st.markdown(f'<div class="dots">{items}</div>', unsafe_allow_html=True)


def ticker(lr, m):
    preds = "".join(
        f"<span>CGPA <b>{c:.2f}</b> → <b>{predict(lr, c):.2f}</b> LPA</span>"
        for c in (5.0, 6.0, 7.0, 8.0, 9.0)
    )
    stats = (
        f"<span>R² <b>{m['r2']:.4f}</b></span>"
        f"<span>MAE <b>{m['mae']:.4f}</b></span>"
        f"<span>RMSE <b>{m['rmse']:.4f}</b></span>"
        f"<span>MAPE <b>{m['mape'] * 100:.2f}%</b></span>"
        f"<span>ADJ R² <b>{m['adj_r2']:.4f}</b></span>"
        f"<span>EXPL. VAR <b>{m['evs']:.4f}</b></span>"
    )
    items = preds + stats
    st.markdown(
        f'<div class="ticker"><div class="ticker-track">{items}{items}</div></div>',
        unsafe_allow_html=True,
    )


def section(text, delay=0.0, scroll=False):
    cls = "section-label rv-s" if scroll else "section-label rv"
    st.markdown(f'<div class="{cls}" style="--delay:{delay}s">{text}</div>', unsafe_allow_html=True)


def split_value(value, decimals):
    scale = 10**decimals
    return divmod(round(value * scale), scale)


def badge(label, value, decimals, delay):
    whole, frac = split_value(value, decimals)
    cls = "p4" if decimals == 4 else "p2"
    return (
        f'<div class="bd" style="--delay:{delay}s"><small>{label}</small>'
        f'<div class="bd-val {cls}" style="--i:{whole};--d:{frac}" role="img" '
        f'aria-label="{label} {value:.4f}"></div></div>'
    )


def input_view(m):
    st.markdown(
        '<div class="hero">'
        '<div class="kicker"><i></i>Simple linear regression · live model</div>'
        '<div class="title"><span class="line"><span>Your CGPA,</span></span>'
        '<span class="line"><span>your <em>package.</em></span></span></div>'
        "<p>Set your CGPA, hit predict, and the model trained on 200 real placement records "
        "tells you what to expect, and where you stand.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    value = _cgpa_input(
        initial=float(st.session_state.cgpa),
        key=f"cgpa_input_{st.session_state.visit}",
        default=None,
    )
    if value and value.get("ts") != st.session_state.last_ts:
        st.session_state.last_ts = value["ts"]
        st.session_state.cgpa = float(value["cgpa"])
        st.session_state.page = "result"
        st.rerun()
    st.markdown(
        '<div class="steps">'
        '<div class="stp" style="--delay:0.5s"><b>01</b><span>Enter your CGPA</span></div>'
        '<div class="stp" style="--delay:0.65s"><b>02</b><span>Hit predict</span></div>'
        '<div class="stp" style="--delay:0.8s"><b>03</b><span>Explore where you stand</span></div>'
        "</div>"
        '<div class="badges">'
        + badge("R² score", m["r2"], 4, 0.7)
        + badge("MAE (LPA)", m["mae"], 4, 0.8)
        + badge("RMSE (LPA)", m["rmse"], 4, 0.9)
        + badge("MAPE", m["mape"] * 100, 2, 1.0)
        + "</div>",
        unsafe_allow_html=True,
    )


def ring(pct, title, sub, delay):
    circ = 339.292
    return (
        f'<div class="ring rv" style="--delay:{delay}s">'
        '<div class="ring-box">'
        '<svg viewBox="0 0 120 120"><circle class="track" cx="60" cy="60" r="54"/>'
        f'<circle class="arc" cx="60" cy="60" r="54" style="stroke-dashoffset:{circ * (1 - pct / 100):.2f}"/></svg>'
        f'<div class="ring-num" style="--p:{round(pct)}" role="img" aria-label="{round(pct)} percent"></div>'
        "</div>"
        f'<div class="ring-title">{title}</div><div class="ring-sub">{sub}</div></div>'
    )


def mcard(name, value, direction, desc, foot, delay, decimals=4, bar=None):
    whole, frac = split_value(value, decimals)
    cls = "p4" if decimals == 4 else "p2"
    bar_html = f'<div class="mc-bar"><i style="width:{min(max(bar, 0), 100):.0f}%"></i></div>' if bar is not None else ""
    return (
        f'<div class="mc rv-s" style="--delay:{delay}s">'
        f'<div class="mc-top"><small>{name}</small><em>{direction}</em></div>'
        f'<div class="mc-val {cls}" style="--i:{whole};--d:{frac}" role="img" aria-label="{name} {value:.4f}"></div>'
        f"<p>{desc}</p>{bar_html}<div class='mc-foot'>{foot}</div></div>"
    )


def dcard(name, value, desc, ok, delay):
    tag = '<em class="ok">✓ pass</em>' if ok else '<em class="warn">! check</em>'
    return (
        f'<div class="dg rv-s" style="--delay:{delay}s"><small>{name}</small>'
        f"<b>{value}</b><span>{desc}</span>{tag}</div>"
    )


def plot(fig, caption):
    st.markdown(f'<div class="cap">{caption}</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def accuracy_lab(df, a):
    t, tr, base, gain = a["test"], a["train"], a["base"], a["gain"]
    overfit = tr["r2"] - t["r2"] > 0.05
    normal = a["shapiro_p"] > 0.05
    spread = abs(a["hetero_corr"]) < 0.3
    notes = []
    notes.append("no sign of overfitting" if not overfit else "a train-test gap worth watching")
    if normal and spread:
        notes.append("residuals pass the normality and constant-spread checks")
    cv_mean = float(a["cv"].mean())

    section("Model accuracy lab", 1.0, scroll=True)
    st.markdown(
        '<div class="verdict rv-s" style="--delay:1s">'
        f'<div class="vd-score">{t["r2"] * 100:.0f}<small>%</small></div>'
        '<div class="vd-body"><div class="vd-title">What the seven regression metrics say</div>'
        f'<p>CGPA explains <b>{t["r2"] * 100:.1f}%</b> of the variation in package. A typical miss is '
        f'<b>{t["mae"]:.2f} LPA</b> (<b>{t["mape"] * 100:.1f}%</b>), which is <b>{gain["mae"]:.0f}% better</b> '
        f"than always guessing the average. Train and test R² differ by {abs(tr['r2'] - t['r2']):.4f}, "
        f"5-fold cross-validation averages <b>{cv_mean:.3f}</b>, and the evidence shows {'; '.join(notes)}.</p></div></div>",
        unsafe_allow_html=True,
    )

    tab_err, tab_fit, tab_res = st.tabs(["Error metrics", "Fit metrics", "Residual diagnostics"])

    with tab_err:
        cards = "".join(
            [
                mcard("MAE", t["mae"], "lower is better", "Average miss in LPA, ignoring direction.",
                      f"{gain['mae']:.0f}% lower than guessing the average ({base['mae']:.3f})", 0.0, bar=gain["mae"]),
                mcard("MSE", t["mse"], "lower is better", "Average squared miss. Big errors count extra.",
                      f"{gain['mse']:.0f}% lower than guessing the average ({base['mse']:.3f})", 0.1, bar=gain["mse"]),
                mcard("RMSE", t["rmse"], "lower is better", "Typical miss in LPA, leaning toward the big ones.",
                      f"{gain['rmse']:.0f}% lower than guessing the average ({base['rmse']:.3f})", 0.2, bar=gain["rmse"]),
                mcard("MAPE", t["mape"] * 100, "lower is better", "Average miss as a percentage of the real package.",
                      f"{gain['mape']:.0f}% lower than guessing the average ({base['mape'] * 100:.1f}%)", 0.3,
                      decimals=2, bar=gain["mape"]),
            ]
        )
        st.markdown(f'<div class="mc-grid">{cards}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            plot(charts.abs_error_lollipop(a), "Every test student's miss, smallest to largest")
        with c2:
            plot(charts.mape_hist(a), "How big the percentage errors are · green is at or under the MAPE")
        c3, c4 = st.columns(2, gap="medium")
        with c3:
            plot(charts.error_by_band(a), "MAE by CGPA band · labels show MAPE and test students")
        with c4:
            plot(charts.baseline_gain(a), "Improvement over a model that always predicts the average")

    with tab_fit:
        cards = "".join(
            [
                mcard("R² score", t["r2"], "higher is better", "Share of package variation that CGPA explains.",
                      f"train {tr['r2']:.4f} · gap {t['r2'] - tr['r2']:+.4f}", 0.0, bar=t["r2"] * 100),
                mcard("Adjusted R²", t["adj_r2"], "higher is better", "R² corrected for the number of features used.",
                      f"sits {t['r2'] - t['adj_r2']:.4f} under R² with one feature", 0.1, bar=t["adj_r2"] * 100),
                mcard("Explained variance", t["evs"], "higher is better", "Like R² but ignores any constant bias.",
                      f"EVS − R² = {t['evs'] - t['r2']:+.4f}, so bias is tiny", 0.2, bar=t["evs"] * 100),
            ]
        )
        st.markdown(f'<div class="mc-grid three">{cards}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            plot(charts.metric_radar(a, df), "Accuracy radar · train against test, 1.0 is perfect")
        with c2:
            plot(charts.actual_vs_predicted(a), "Actual against predicted · colour shows size of miss, band is ±RMSE")
        c3, c4 = st.columns(2, gap="medium")
        with c3:
            plot(charts.cv_scores(a), "5-fold cross-validation R² · does the score hold on other splits?")
        with c4:
            plot(charts.train_test_bars(a), "Train against test · matching bars mean no overfitting")

    with tab_res:
        cards = "".join(
            [
                dcard("Bias", f"{a['bias']:+.3f} LPA", "Average residual. Close to zero means no systematic over or under prediction.",
                      abs(a["bias"]) < 0.1, 0.0),
                dcard("Skewness", f"{a['skew']:+.3f}", "Zero means errors are symmetric around the line.",
                      abs(a["skew"]) < 0.5, 0.1),
                dcard("Normality", f"p = {a['shapiro_p']:.3f}", "Shapiro-Wilk test. Above 0.05 means the errors look normal.",
                      normal, 0.2),
                dcard("Constant spread", f"r = {a['hetero_corr']:+.3f}", "Correlation of error size with fitted value. Near zero means no fanning out.",
                      spread, 0.3),
            ]
        )
        st.markdown(f'<div class="mc-grid">{cards}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            plot(charts.residual_vs_fitted(a), "Residuals against fitted · a random cloud means a good linear fit")
        with c2:
            plot(charts.qq_plot(a), "Normal Q-Q plot · points on the dashed line mean normal errors")
        c3, c4 = st.columns(2, gap="medium")
        with c3:
            plot(charts.residual_hist(a), "Residual histogram with a fitted normal curve")
        with c4:
            st.markdown(
                '<div class="cap">Reading the diagnostics</div>'
                '<div class="read rv-s" style="--delay:0.4s">'
                "<p><b>Why they matter.</b> R² and the error metrics say how close the predictions are. "
                "The diagnostics say whether the straight line is the right shape.</p>"
                f"<p><b>Bias {a['bias']:+.3f}.</b> Predictions run about {abs(a['bias']):.2f} LPA "
                f"{'high' if a['bias'] < 0 else 'low'} on average, which is negligible against a typical miss of {t['mae']:.2f}.</p>"
                f"<p><b>Within {0.25} LPA:</b> {a['within_025']:.0f}% of test students. "
                f"<b>Within {0.5} LPA:</b> {a['within_050']:.1f}%. Worst miss: {a['max_err']:.2f} LPA.</p></div>",
                unsafe_allow_html=True,
            )


def result_view(df, lr, ana):
    metrics = ana["test"]
    cgpa = float(st.session_state.cgpa)
    ins = build_insights(df, lr, cgpa, metrics)
    shown = ins["shown"]
    n = ins["n"]
    total = round(shown * 100)
    whole, frac = divmod(total, 100)
    pkg_min = float(df["package"].min())
    pkg_max = float(df["package"].max())
    fill = min(max((shown - pkg_min) / (pkg_max - pkg_min), 0.0), 1.0) * 100

    st.markdown(
        '<div class="calc"><div class="calc-inner"><div class="calc-ring"></div>'
        '<div class="calc-steps"><span>Reading 200 placement records</span>'
        "<span>Fitting the regression line</span><span>Computing your package</span></div>"
        '<div class="calc-bar"><i></i></div></div></div>',
        unsafe_allow_html=True,
    )

    top_l, top_r = st.columns([2, 5], vertical_alignment="center")
    with top_l:
        st.button("← Recalculate", key="back_top", on_click=go_back, type="secondary")
    with top_r:
        st.markdown(
            f'<div class="used rv" style="--delay:0s">Using CGPA <b>{cgpa:.2f}</b></div>',
            unsafe_allow_html=True,
        )

    note = ""
    if cgpa < float(df["cgpa"].min()) or cgpa > float(df["cgpa"].max()):
        note = (
            f'<div class="note">The training data only covers CGPAs from {df["cgpa"].min():.2f} to '
            f'{df["cgpa"].max():.2f}, so this estimate is an extrapolation and less reliable.</div>'
        )

    st.markdown(
        '<div class="hero-grid">'
        '<div class="result rv" style="--delay:0.05s">'
        '<div class="result-label">Your predicted package</div>'
        f'<div class="figure"><div class="num" style="--i:{whole};--d:{frac}" role="img" '
        f'aria-label="{shown:.2f} lakhs per annum"></div><div class="unit">LPA</div></div>'
        f'<div class="range">Likely range <b>{ins["low"]:.2f}</b> to <b>{ins["high"]:.2f}</b> LPA'
        f"<span>typical error ±{metrics['rmse']:.2f}</span>"
        f"<span>about {metrics['mape'] * 100:.1f}% off</span></div>"
        f'<div class="meter"><div class="meter-fill" style="width:{fill:.1f}%"></div></div>'
        f'<div class="meter-scale"><span>{pkg_min:.2f} lowest</span><span>{pkg_max:.2f} highest</span></div>'
        f"{note}</div>"
        '<div class="rings">'
        + ring(ins["cgpa_pct"], "ahead on CGPA", f"better than {ins['cgpa_ahead']} of {n} students", 0.15)
        + ring(ins["pkg_pct"], "ahead on package", f"higher than {ins['pkg_ahead']} of {n} packages", 0.3)
        + "</div></div>",
        unsafe_allow_html=True,
    )

    section("Where you stand among 200 students", 0.35)
    sorted_pk = sorted(df["package"].tolist())
    ahead = ins["pkg_ahead"]
    cells = []
    for i in range(n):
        cls = "d on" if i < ahead else "d"
        cells.append(f'<i class="{cls}" style="--k:{i}"></i>')
    cells.insert(ahead, f'<i class="d me" style="--k:{ahead}"></i>')
    st.markdown(
        '<div class="crowd rv" style="--delay:0.4s">'
        f'<div class="crowd-head"><div class="crowd-count" style="--p:{ahead}"></div>'
        f'<div class="crowd-text">of {n} students received a lower package than you are predicted to get.</div></div>'
        f'<div class="grid">{"".join(cells)}</div>'
        '<div class="legend"><span><i class="d on"></i>lower than you</span>'
        '<span><i class="d"></i>higher than you</span><span><i class="d me"></i>you</span></div></div>',
        unsafe_allow_html=True,
    )

    peers = ins["peers"]
    if peers["n"]:
        diff = shown - peers["avg"]
        peer_card = (
            f'<div class="ins rv" style="--delay:0.5s"><b>{peers["avg"]:.2f}<small>LPA</small></b>'
            f"<span>average for the {peers['n']} students within ±0.5 CGPA of you. "
            f'You are predicted <em>{diff:+.2f}</em> against them.</span></div>'
            f'<div class="ins rv" style="--delay:0.6s"><b>{peers["min"]:.2f} to {peers["max"]:.2f}</b>'
            "<span>the real package range among those same students.</span></div>"
        )
    else:
        peer_card = (
            '<div class="ins rv" style="--delay:0.5s"><b>n/a</b>'
            "<span>no students in the data are within ±0.5 CGPA of yours.</span></div>"
            '<div class="ins rv" style="--delay:0.6s"><b>n/a</b><span>no peer range to show.</span></div>'
        )
    st.markdown(
        '<div class="ins-grid">'
        f'<div class="ins rv" style="--delay:0.4s"><b>#{ins["rank"]}</b>'
        f"<span>your rank by CGPA out of {n} students.</span></div>"
        + peer_card
        + f'<div class="ins rv" style="--delay:0.7s"><b>+{ins["per_tenth"]:.3f}<small>LPA</small></b>'
        "<span>the model adds this for every extra 0.1 of CGPA.</span></div></div>",
        unsafe_allow_html=True,
    )

    section("What if your CGPA were higher?", 0.5, scroll=True)
    if ins["what_if"]:
        max_gain = max(w["gain"] for w in ins["what_if"]) or 1.0
        cards = "".join(
            f'<div class="wi rv-s" style="--delay:{0.55 + i * 0.1}s">'
            f'<small>+{w["delta"]:.2f} CGPA</small>'
            f'<div class="wi-cgpa">{w["cgpa"]:.2f}</div>'
            f'<div class="wi-pred">{w["pred"]:.2f}<span>LPA</span></div>'
            f'<div class="wi-bar"><i style="width:{max(w["gain"] / max_gain, 0.05) * 100:.0f}%"></i></div>'
            f'<div class="wi-gain">{w["gain"]:+.2f} LPA</div></div>'
            for i, w in enumerate(ins["what_if"])
        )
        st.markdown(f'<div class="wi-grid">{cards}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="note">Your CGPA is already at the top of the scale.</div>',
            unsafe_allow_html=True,
        )

    section("Where the numbers come from", 0.6, scroll=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="cap">Package distribution · green bars are packages below yours</div>', unsafe_allow_html=True)
        st.plotly_chart(charts.package_distribution(df, shown), width="stretch", config={"displayModeBar": False})
    with c2:
        st.markdown('<div class="cap">CGPA distribution · green bars are CGPAs below yours</div>', unsafe_allow_html=True)
        st.plotly_chart(charts.cgpa_distribution(df, cgpa), width="stretch", config={"displayModeBar": False})

    section("The fit, with the typical error band", 0.7, scroll=True)
    st.plotly_chart(
        charts.fit_band(df, lr, cgpa, shown, metrics["rmse"]),
        width="stretch",
        config={"displayModeBar": False},
    )

    section("Density and percentile", 0.8, scroll=True)
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        st.markdown('<div class="cap">Where students cluster · brighter means more students</div>', unsafe_allow_html=True)
        st.plotly_chart(charts.density(df, cgpa, shown), width="stretch", config={"displayModeBar": False})
    with c4:
        st.markdown('<div class="cap">Percentile curve · share of students at or below each package</div>', unsafe_allow_html=True)
        st.plotly_chart(charts.ecdf(df, shown), width="stretch", config={"displayModeBar": False})

    section("Package wheel and targets", 0.9, scroll=True)
    c5, c6 = st.columns([5, 6], gap="medium")
    with c5:
        st.markdown('<div class="cap">Average real package by CGPA band · orange dot is you</div>', unsafe_allow_html=True)
        st.plotly_chart(
            charts.band_wheel(ins["bands"], ins["band"], shown),
            width="stretch",
            config={"displayModeBar": False},
        )
    with c6:
        rows = []
        for i, t in enumerate(ins["targets"]):
            ratio = min(cgpa / t["need"], 1.0) * 100 if t["need"] > 0 else 100
            if t["reached"]:
                status = '<em class="ok">✓ already on track</em>'
            elif t["possible"]:
                status = f'<em>+{t["gap"]:.2f} CGPA to go</em>'
            else:
                status = "<em>needs a CGPA above 10</em>"
            need = f'{t["need"]:.2f}' if t["possible"] else "10+"
            rows.append(
                f'<div class="tg rv-s" style="--delay:{0.95 + i * 0.1}s">'
                f'<div class="tg-top"><b>{t["target"]:.1f} LPA</b><span>needs CGPA {need}</span></div>'
                f'<div class="tg-bar"><i style="width:{ratio:.0f}%"></i></div>{status}</div>'
            )
        st.markdown(
            '<div class="cap">CGPA the model says you need for each target</div>'
            f'<div class="tg-list">{"".join(rows)}</div>',
            unsafe_allow_html=True,
        )

    accuracy_lab(df, ana)

    sign = "−" if ins["intercept"] < 0 else "+"
    st.markdown(
        '<div class="math rv-s" style="--delay:1.05s">'
        f'<span class="eq">package = <em>{ins["slope"]:.4f}</em> × CGPA {sign} <u>{abs(ins["intercept"]):.4f}</u></span>'
        f'<span class="step">plug in your CGPA → <b>{ins["slope"]:.4f} × {cgpa:.2f} {sign} '
        f'{abs(ins["intercept"]):.4f} = {ins["raw"]:.2f}</b> LPA</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="again">Try a different CGPA and see how the picture changes.</div>', unsafe_allow_html=True)
    st.button("← Recalculate with a new CGPA", key="back_bottom", on_click=go_back, type="primary")

    st.markdown(
        '<div class="foot"><span>scikit-learn · LinearRegression</span>'
        "<span>test_size 0.2 · random_state 2</span>"
        f"<span>{n} records</span></div>",
        unsafe_allow_html=True,
    )
