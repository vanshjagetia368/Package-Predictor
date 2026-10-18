from model import predict

BANDS = [
    (0.0, 6.0, "<6"),
    (6.0, 7.0, "6-7"),
    (7.0, 8.0, "7-8"),
    (8.0, 9.0, "8-9"),
    (9.0, 10.01, "9+"),
]

TARGETS = (3.0, 3.5, 4.0, 4.5)
WHAT_IF = (0.25, 0.5, 1.0)


def band_index(cgpa):
    for i, (lo, hi, _) in enumerate(BANDS):
        if lo <= cgpa < hi:
            return i
    return len(BANDS) - 1


def build(df, lr, cgpa, metrics):
    raw = predict(lr, cgpa)
    shown = max(raw, 0.0)
    slope = float(lr.coef_[0])
    intercept = float(lr.intercept_)
    n = len(df)

    cgpa_ahead = int((df["cgpa"] < cgpa).sum())
    pkg_ahead = int((df["package"] < shown).sum())
    rank = int((df["cgpa"] > cgpa).sum()) + 1

    window = df[(df["cgpa"] >= cgpa - 0.5) & (df["cgpa"] <= cgpa + 0.5)]
    peers = {
        "n": len(window),
        "avg": float(window["package"].mean()) if len(window) else None,
        "min": float(window["package"].min()) if len(window) else None,
        "max": float(window["package"].max()) if len(window) else None,
    }

    bands = []
    for lo, hi, label in BANDS:
        sub = df[(df["cgpa"] >= lo) & (df["cgpa"] < hi)]
        bands.append(
            {
                "label": label,
                "n": len(sub),
                "avg": float(sub["package"].mean()) if len(sub) else 0.0,
            }
        )

    what_if = []
    for delta in WHAT_IF:
        c2 = cgpa + delta
        if c2 <= 10:
            p2 = max(predict(lr, c2), 0.0)
            what_if.append({"delta": delta, "cgpa": c2, "pred": p2, "gain": p2 - shown})

    targets = []
    for t in TARGETS:
        need = (t - intercept) / slope
        targets.append(
            {
                "target": t,
                "need": need,
                "reached": shown >= t,
                "possible": need <= 10,
                "gap": max(need - cgpa, 0.0),
            }
        )

    return {
        "cgpa": cgpa,
        "raw": raw,
        "shown": shown,
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "cgpa_ahead": cgpa_ahead,
        "cgpa_pct": cgpa_ahead / n * 100,
        "pkg_ahead": pkg_ahead,
        "pkg_pct": pkg_ahead / n * 100,
        "rank": rank,
        "peers": peers,
        "bands": bands,
        "band": band_index(cgpa),
        "what_if": what_if,
        "targets": targets,
        "low": max(shown - metrics["rmse"], 0.0),
        "high": shown + metrics["rmse"],
        "per_tenth": slope * 0.1,
    }
