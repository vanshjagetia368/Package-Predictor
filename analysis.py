import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from insights import BANDS
from model import evaluate


def baseline(y_train, x_test, y_test):
    class Mean:
        def __init__(self, value):
            self.value = value

        def predict(self, x):
            return np.full(len(x), self.value)

    return evaluate(Mean(float(y_train.mean())), x_test, y_test)


def build(df, lr, x_train, x_test, y_train, y_test):
    test = evaluate(lr, x_test, y_test)
    train = evaluate(lr, x_train, y_train)
    base = baseline(y_train, x_test, y_test)

    pred = lr.predict(x_test)
    actual = y_test.values
    resid = actual - pred
    ape = np.abs(resid) / np.abs(actual) * 100

    skew = float(stats.skew(resid))
    kurt = float(stats.kurtosis(resid))
    shapiro = stats.shapiro(resid)
    corr_abs = float(np.corrcoef(np.abs(resid), pred)[0, 1])

    x_all = df.iloc[:, 0:1]
    y_all = df.iloc[:, -1]
    cv = cross_val_score(
        LinearRegression(),
        x_all,
        y_all,
        cv=KFold(n_splits=5, shuffle=True, random_state=2),
        scoring="r2",
    )

    bands = []
    for lo, hi, label in BANDS:
        mask = (x_test.iloc[:, 0].values >= lo) & (x_test.iloc[:, 0].values < hi)
        if mask.any():
            bands.append(
                {
                    "label": label,
                    "n": int(mask.sum()),
                    "mae": float(np.abs(resid[mask]).mean()),
                    "mape": float(ape[mask].mean()),
                }
            )
        else:
            bands.append({"label": label, "n": 0, "mae": 0.0, "mape": 0.0})

    gain = {
        key: (base[key] - test[key]) / base[key] * 100 for key in ("mae", "mse", "rmse", "mape")
    }

    return {
        "test": test,
        "train": train,
        "base": base,
        "gain": gain,
        "pred": pred,
        "actual": actual,
        "cgpa": x_test.iloc[:, 0].values,
        "resid": resid,
        "ape": ape,
        "skew": skew,
        "kurt": kurt,
        "shapiro_p": float(shapiro.pvalue),
        "hetero_corr": corr_abs,
        "cv": cv,
        "bands": bands,
        "within_025": float((np.abs(resid) <= 0.25).mean() * 100),
        "within_050": float((np.abs(resid) <= 0.5).mean() * 100),
        "max_err": float(np.abs(resid).max()),
        "bias": float(resid.mean()),
    }
