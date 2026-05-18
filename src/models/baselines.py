"""Baseline models for HantaCast evaluation.

Statistical (6): naive, mean, drift, moving_average, exp_smoothing, TheilSen
Linear/Regularized (3): Ridge, Lasso, ElasticNet
Kernel/KNN (3): SVR_RBF, SVR_Linear, KNN
Gaussian Process (1): GaussianProcess
Tree (2): DecisionTree, ExtraTrees
Deep (2): MLP, RNN
Mechanistic (1): SEIRD (no MixLinear signal)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as GPRBF
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.data.dataset import make_supervised_windows, select_feature_columns
from src.models.seird_dynamics import SEIRDDynamics
from src.utils.seed import set_global_seed

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    ExponentialSmoothing = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None

log = logging.getLogger(__name__)


def _make_supervised(df: pd.DataFrame, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build supervised arrays from dataframe."""
    feature_columns = select_feature_columns(df)
    dataset = make_supervised_windows(df, lookback, horizon, target_col="new_cases", feature_columns=feature_columns)
    return np.asarray(dataset["X"], dtype=float), np.asarray(dataset["y"], dtype=float), feature_columns


def _predict_frame(preds: np.ndarray, df: pd.DataFrame, future_covariates: pd.DataFrame | None, model_name: str) -> pd.DataFrame:
    """Build prediction DataFrame compatible with evaluate_hantacast."""
    horizon = len(preds)
    if future_covariates is not None and "date" in future_covariates.columns:
        dates = pd.to_datetime(future_covariates["date"]).tolist()[:horizon]
    else:
        last_date = pd.to_datetime(df["date"]).iloc[-1]
        dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D").tolist()

    last_cumulative = float(pd.to_numeric(df["cumulative_cases"], errors="coerce").iloc[-1]) if "cumulative_cases" in df.columns else 0.0
    p = np.maximum(np.asarray(preds, dtype=float), 0.0)
    return pd.DataFrame({
        "model": model_name,
        "date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
        "forecast_median": p.astype(float),
        "forecast_lower": np.maximum(p - 2.0, 0.0),
        "forecast_upper": p + 2.0,
        "interval_type": "residual_se",
        "interval_source": f"{model_name}_approx",
        "cumulative_cases_pred": last_cumulative + np.cumsum(p),
        "active_cases_pred": np.repeat(np.nan, horizon),
        "n_train": len(df),
    })


# ═══════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════

class _Baseline(ABC):
    """Common interface for all baselines."""

    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "_Baseline":
        self.train_df = df.copy()
        self._fit_impl(df)
        self._fitted = True
        return self

    @abstractmethod
    def _fit_impl(self, df: pd.DataFrame): ...

    def predict(self, horizon: int | None = None, future_covariates: pd.DataFrame | None = None) -> pd.DataFrame:
        horizon = int(horizon or 1)
        preds = self._predict_impl(horizon, future_covariates)
        preds = np.asarray(preds, dtype=float).ravel()[:horizon]
        return _predict_frame(preds, self.train_df, future_covariates, self.name)

    @abstractmethod
    def _predict_impl(self, horizon: int, future_covariates: pd.DataFrame | None) -> np.ndarray: ...


# ═══════════════════════════════════════════════════════════════
# Statistical — naive family
# ═══════════════════════════════════════════════════════════════

class NaiveBaseline(_Baseline):
    def __init__(self, random_state=42):
        super().__init__("naive", random_state)

    def _fit_impl(self, df):
        series = pd.to_numeric(df["new_cases"], errors="coerce").fillna(0.0)
        self.last_value = float(series.iloc[-1])

    def _predict_impl(self, horizon, future_covariates):
        return np.repeat(self.last_value, horizon)


class MeanBaseline(_Baseline):
    def __init__(self, random_state=42):
        super().__init__("mean", random_state)

    def _fit_impl(self, df):
        series = pd.to_numeric(df["new_cases"], errors="coerce").fillna(0.0)
        self.mean_value = float(series.mean())

    def _predict_impl(self, horizon, future_covariates):
        return np.repeat(self.mean_value, horizon)


class DriftBaseline(_Baseline):
    def __init__(self, random_state=42):
        super().__init__("drift", random_state)

    def _fit_impl(self, df):
        series = pd.to_numeric(df["new_cases"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        self.last = float(series[-1])
        diffs = np.diff(series)
        self.drift = float(np.mean(diffs)) if len(diffs) > 0 else 0.0

    def _predict_impl(self, horizon, future_covariates):
        return np.array([max(self.last + self.drift * (i + 1), 0.0) for i in range(horizon)])


# ═══════════════════════════════════════════════════════════════
# Statistical — smoothing
# ═══════════════════════════════════════════════════════════════

class MovingAverageBaseline(_Baseline):
    def __init__(self, lookback=3, random_state=42):
        super().__init__("moving_average", random_state)
        self.lookback = lookback

    def _fit_impl(self, df):
        series = pd.to_numeric(df["new_cases"], errors="coerce").fillna(0.0)
        window = series.tail(self.lookback)
        self.avg = float(window.mean()) if len(window) > 0 else 0.0

    def _predict_impl(self, horizon, future_covariates):
        return np.repeat(self.avg, horizon)


class ExpSmoothingBaseline(_Baseline):
    def __init__(self, random_state=42):
        super().__init__("exp_smoothing", random_state)

    def _fit_impl(self, df):
        self.series = pd.to_numeric(df["new_cases"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        self.model = None
        if ExponentialSmoothing is not None and len(self.series) >= 3:
            try:
                self.model = ExponentialSmoothing(
                    self.series, trend="add", damped_trend=True, initialization_method="estimated"
                ).fit(optimized=True)
            except Exception:
                self.model = None

    def _predict_impl(self, horizon, future_covariates):
        if self.model is not None:
            try:
                return np.asarray(self.model.forecast(horizon), dtype=float)
            except Exception:
                pass
        return np.repeat(float(self.series[-1]), horizon)


# ═══════════════════════════════════════════════════════════════
# Linear / Regularized regression
# ═══════════════════════════════════════════════════════════════

class _SklearnRegressorBaseline(_Baseline):
    """Base for sklearn regressors — autoregression on new_cases only."""

    def __init__(self, name, model, lookback=3, horizon=1, random_state=42):
        super().__init__(name, random_state)
        self.lookback = lookback
        self.horizon = horizon
        self.model = model

    def _fit_impl(self, df):
        series = pd.to_numeric(df["new_cases"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        X, y = [], []
        for i in range(self.lookback, len(series)):
            X.append(series[i - self.lookback : i])
            y.append(series[i])
        if len(X) < 2:
            self.model = None
            self.last_vals = series[-self.lookback:].copy()
            return
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float)
        try:
            self.model.fit(Xa, ya)
        except Exception:
            self.model = None
        self.last_vals = series[-self.lookback:].copy()

    def _predict_impl(self, horizon, future_covariates):
        if self.model is None:
            return np.repeat(float(self.last_vals[-1]) if len(self.last_vals) > 0 else 0.0, horizon)
        history = list(self.last_vals.astype(float))
        preds = []
        for _ in range(horizon):
            x = np.asarray(history[-self.lookback:], dtype=float).reshape(1, -1)
            p = float(self.model.predict(x)[0])
            p = max(p, 0.0)
            preds.append(p)
            history.append(p)
        return np.asarray(preds, dtype=float)


class LinearRegressionBaseline(_SklearnRegressorBaseline):
    def __init__(self, lookback=3, horizon=1, random_state=42):
        super().__init__("LinearRegression", LinearRegression(), lookback, horizon, random_state)


class TheilSenBaseline(_SklearnRegressorBaseline):
    def __init__(self, lookback=3, horizon=1, random_state=42):
        super().__init__("TheilSen", TheilSenRegressor(random_state=random_state, max_subpopulation=100), lookback, horizon, random_state)


class RidgeBaseline(_SklearnRegressorBaseline):
    def __init__(self, lookback=3, horizon=1, random_state=42):
        super().__init__("Ridge", Ridge(alpha=1.0, random_state=random_state), lookback, horizon, random_state)


class LassoBaseline(_SklearnRegressorBaseline):
    def __init__(self, lookback=3, horizon=1, random_state=42):
        super().__init__("Lasso", Lasso(alpha=0.1, random_state=random_state, max_iter=5000), lookback, horizon, random_state)


class ElasticNetBaseline(_SklearnRegressorBaseline):
    def __init__(self, lookback=3, horizon=1, random_state=42):
        super().__init__("ElasticNet", ElasticNet(alpha=0.2, l1_ratio=0.5, random_state=random_state, max_iter=5000), lookback, horizon, random_state)


# ═══════════════════════════════════════════════════════════════
# SVR family
# ═══════════════════════════════════════════════════════════════

class SVRRBFBaseline(_SklearnRegressorBaseline):
    def __init__(self, lookback=3, horizon=1, random_state=42):
        super().__init__("SVR_RBF", SVR(kernel="rbf", C=1.0, epsilon=0.1), lookback, horizon, random_state)


class SVRLinearBaseline(_SklearnRegressorBaseline):
    def __init__(self, lookback=3, horizon=1, random_state=42):
        super().__init__("SVR_Linear", SVR(kernel="linear", C=1.0, epsilon=0.1), lookback, horizon, random_state)


# ═══════════════════════════════════════════════════════════════
# KNN
# ═══════════════════════════════════════════════════════════════

class KNNBaseline(_SklearnRegressorBaseline):
    def __init__(self, lookback=3, horizon=1, random_state=42):
        super().__init__("KNN", KNeighborsRegressor(n_neighbors=3), lookback, horizon, random_state)


# ═══════════════════════════════════════════════════════════════
# Gaussian Process
# ═══════════════════════════════════════════════════════════════

class GaussianProcessBaseline(_SklearnRegressorBaseline):
    def __init__(self, lookback=3, horizon=1, random_state=42):
        super().__init__(
            "GaussianProcess",
            GaussianProcessRegressor(kernel=GPRBF(), random_state=random_state, normalize_y=True),
            lookback, horizon, random_state,
        )


# ═══════════════════════════════════════════════════════════════
# Tree models
# ═══════════════════════════════════════════════════════════════

class DecisionTreeBaseline(_SklearnRegressorBaseline):
    def __init__(self, lookback=3, horizon=1, random_state=42):
        super().__init__(
            "DecisionTree",
            DecisionTreeRegressor(max_depth=3, random_state=random_state),
            lookback, horizon, random_state,
        )


class ExtraTreesBaseline(_SklearnRegressorBaseline):
    def __init__(self, lookback=3, horizon=1, random_state=42):
        super().__init__(
            "ExtraTrees",
            ExtraTreesRegressor(n_estimators=10, max_depth=4, random_state=random_state),
            lookback, horizon, random_state,
        )


# ═══════════════════════════════════════════════════════════════
# Deep learning — MLP (torch)
# ═══════════════════════════════════════════════════════════════

class MLPBaseline(_Baseline):
    def __init__(self, lookback=3, horizon=1, random_state=42, epochs=80, lr=0.01, wd=1e-3):
        super().__init__("MLP", random_state)
        self.lookback = lookback
        self.horizon = horizon
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.device = torch.device("cuda" if torch is not None and torch.cuda.is_available() else "cpu") if torch is not None else None
        self.model: Optional[nn.Module] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.feature_columns: list[str] = []

    def _fit_impl(self, df):
        set_global_seed(self.random_state)
        X, y, self.feature_columns = _make_supervised(df, self.lookback, self.horizon)
        if len(X) < 1 or torch is None:
            self.model = None
            return
        n_features = X.shape[-1]
        n_samples = X.shape[0]
        X2d = X.reshape(n_samples, -1)
        y1d = y[:, 0] if y.ndim > 1 else y.ravel()[:n_samples]

        self.mean_ = X2d.mean(axis=0)
        self.std_ = X2d.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        Xn = (X2d - self.mean_) / self.std_

        self.model = nn.Sequential(
            nn.Linear(n_features * self.lookback, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        ).to(self.device)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        Xt = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y1d, dtype=torch.float32, device=self.device).reshape(-1, 1)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=min(8, n_samples), shuffle=False)

        n_epochs = min(self.epochs, 50) if n_samples < 5 else self.epochs
        for _ in range(n_epochs):
            self.model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = nn.MSELoss()(self.model(xb), yb)
                loss.backward()
                opt.step()
        self.model.eval()

        tail = df[self.feature_columns].tail(self.lookback).to_numpy(dtype=float)
        if len(tail) < self.lookback:
            pad = np.tile(tail[0], (self.lookback - len(tail), 1))
            tail = np.vstack([pad, tail])
        self.last_window_norm = ((tail.ravel() - self.mean_) / self.std_).astype(np.float32)

    def _predict_impl(self, horizon, future_covariates):
        if self.model is None:
            return np.repeat(0.0, horizon)
        with torch.no_grad():
            xt = torch.tensor(self.last_window_norm.reshape(1, -1), dtype=torch.float32, device=self.device)
            pred = float(self.model(xt).item())
        return np.array([max(pred, 0.0)])


# ═══════════════════════════════════════════════════════════════
# Deep learning — RNN (torch)
# ═══════════════════════════════════════════════════════════════

class _SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size=8):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class RNNBaseline(_Baseline):
    def __init__(self, lookback=3, horizon=1, random_state=42, epochs=80, lr=0.01, wd=1e-3):
        super().__init__("RNN", random_state)
        self.lookback = lookback
        self.horizon = horizon
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.device = torch.device("cuda" if torch is not None and torch.cuda.is_available() else "cpu") if torch is not None else None
        self.model: Optional[nn.Module] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.feature_columns: list[str] = []

    def _fit_impl(self, df):
        set_global_seed(self.random_state)
        X, y, self.feature_columns = _make_supervised(df, self.lookback, self.horizon)
        if len(X) < 1 or torch is None:
            self.model = None
            return
        n_samples, lookback, n_features = X.shape

        self.mean_ = X.reshape(-1, n_features).mean(axis=0)
        self.std_ = X.reshape(-1, n_features).std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        Xn = (X - self.mean_) / self.std_
        y1d = y[:, 0] if y.ndim > 1 else y.ravel()[:n_samples]

        self.model = _SimpleRNN(n_features, hidden_size=8).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        Xt = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y1d, dtype=torch.float32, device=self.device).reshape(-1, 1)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=min(8, n_samples), shuffle=False)

        n_epochs = min(self.epochs, 50) if n_samples < 5 else self.epochs
        for _ in range(n_epochs):
            self.model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = nn.MSELoss()(self.model(xb), yb)
                loss.backward()
                opt.step()
        self.model.eval()

        tail = df[self.feature_columns].tail(self.lookback).to_numpy(dtype=float)
        if len(tail) < self.lookback:
            pad = np.tile(tail[0], (self.lookback - len(tail), 1))
            tail = np.vstack([pad, tail])
        self.last_window_norm = ((tail - self.mean_) / self.std_).astype(np.float32)

    def _predict_impl(self, horizon, future_covariates):
        if self.model is None:
            return np.repeat(0.0, horizon)
        with torch.no_grad():
            xt = torch.tensor(self.last_window_norm[None, :, :], dtype=torch.float32, device=self.device)
            pred = float(self.model(xt).item())
        return np.array([max(pred, 0.0)])


# ═══════════════════════════════════════════════════════════════
# SEIRD (no MixLinear signal)
# ═══════════════════════════════════════════════════════════════

class SEIRDBaseline(_Baseline):
    def __init__(self, random_state=42, param_grid=None):
        super().__init__("SEIRD", random_state)
        self._param_grid = param_grid

    def _fit_impl(self, df):
        self.seird = SEIRDDynamics(
            random_state=self.random_state,
            mixlinear_signal_scale=0.0,  # no MixLinear signal
            param_grid=self._param_grid,
        )
        self.seird.fit(df)

    def _predict_impl(self, horizon, future_covariates):
        if future_covariates is None:
            last = self.train_df.iloc[-1]
            future_covariates = pd.DataFrame({
                "date": pd.date_range(pd.to_datetime(last["date"]) + pd.Timedelta(days=1), periods=horizon, freq="D"),
                "intervention_index": [float(last.get("intervention_index", 0.2))] * horizon,
                "mobility_index": [float(last.get("mobility_index", 0.0))] * horizon,
                "flight_volume": [float(last.get("flight_volume", 0.0))] * horizon,
                "behavior_response_index": [float(last.get("behavior_response_index", 0.25))] * horizon,
            })
        preds_mean, _ = self.seird.simulate_forward(future_covariates)
        return preds_mean


# ═══════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════

def build_all_baselines(lookback=3, horizon=1, random_state=42) -> dict[str, _Baseline]:
    """Build all baseline models."""
    return {
        # Statistical — naive family
        "naive": NaiveBaseline(random_state),
        "mean": MeanBaseline(random_state),
        "drift": DriftBaseline(random_state),
        # Statistical — smoothing
        "moving_average": MovingAverageBaseline(lookback, random_state),
        "exp_smoothing": ExpSmoothingBaseline(random_state),
        # Linear / robust
        "LinearRegression": LinearRegressionBaseline(lookback, horizon, random_state),
        "TheilSen": TheilSenBaseline(lookback, horizon, random_state),
        # Regularized
        "Ridge": RidgeBaseline(lookback, horizon, random_state),
        "Lasso": LassoBaseline(lookback, horizon, random_state),
        "ElasticNet": ElasticNetBaseline(lookback, horizon, random_state),
        # SVR
        "SVR_RBF": SVRRBFBaseline(lookback, horizon, random_state),
        "SVR_Linear": SVRLinearBaseline(lookback, horizon, random_state),
        # KNN
        "KNN": KNNBaseline(lookback, horizon, random_state),
        # Gaussian Process
        "GaussianProcess": GaussianProcessBaseline(lookback, horizon, random_state),
        # Tree
        "DecisionTree": DecisionTreeBaseline(lookback, horizon, random_state),
        "ExtraTrees": ExtraTreesBaseline(lookback, horizon, random_state),
        # Deep learning
        "MLP": MLPBaseline(lookback, horizon, random_state),
        "RNN": RNNBaseline(lookback, horizon, random_state),
        # Mechanistic
        "SEIRD": SEIRDBaseline(random_state),
    }
