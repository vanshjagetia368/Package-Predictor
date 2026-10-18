from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "data" / "placement.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def train():
    df = load_data()
    x = df.iloc[:, 0:1]
    y = df.iloc[:, -1]
    lr = LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    lr.fit(x_train, y_train)
    return df, lr, x_train, x_test, y_train, y_test


def predict(lr, cgpa):
    return float(lr.predict(pd.DataFrame({"cgpa": [cgpa]}))[0])


def evaluate(lr, x_test, y_test):
    y_pred = lr.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - ((1 - r2) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1))
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mape": mean_absolute_percentage_error(y_test, y_pred),
        "r2": r2,
        "evs": explained_variance_score(y_test, y_pred),
        "adj_r2": adj_r2,
    }
