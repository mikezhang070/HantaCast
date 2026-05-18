from __future__ import annotations

import copy
import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data.dataset import make_supervised_windows, select_feature_columns
from src.utils.seed import set_global_seed

log = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

_TorchModuleBase = nn.Module if nn is not None else object


class _MixLinearNet(_TorchModuleBase):
    """MixLinear temporal signal learner — a gated mixture of experts for time series.

    Four expert branches (temporal, trend, frequency, covariate) are combined
    via a learned gating network. The temporal-only mode provides a linear
    baseline within the mixture.
    """

    def __init__(
        self,
        lookback: int,
        n_features: int,
        horizon: int,
        covariate_indices: list[int],
        dropout: float = 0.1,
        use_trend: bool = True,
        use_frequency: bool = True,
        use_covariate: bool = True,
        temporal_only: bool = False,
    ):
        super().__init__()
        self.lookback = lookback
        self.n_features = n_features
        self.horizon = horizon
        self.covariate_indices = covariate_indices
        self.use_trend = use_trend
        self.use_frequency = use_frequency
        self.use_covariate = use_covariate
        self.temporal_only = temporal_only
        self.freq_dim = min(4, lookback // 2 + 1)
        self.expert_names = ["temporal", "trend", "frequency", "covariate"]

        self.temporal = nn.Linear(lookback * n_features, horizon)
        self.trend = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, horizon))
        self.frequency = nn.Sequential(nn.Linear(self.freq_dim, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, horizon))
        cov_dim = max(1, lookback * max(len(covariate_indices), 1))
        self.covariate = nn.Sequential(nn.Linear(cov_dim, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, horizon))
        self.gating = nn.Sequential(nn.Linear(lookback * n_features, 24), nn.ReLU(), nn.Dropout(dropout), nn.Linear(24, 4))
        self.softplus = nn.Softplus()

    def forward(self, x):
        batch_size = x.shape[0]
        flattened = x.reshape(batch_size, -1)
        case_series = x[:, :, 0]

        trend_features = torch.stack([
            case_series[:, -1],
            case_series.mean(dim=1),
            case_series[:, -1] - case_series[:, 0],
            torch.diff(case_series, dim=1).mean(dim=1) if case_series.shape[1] > 1 else case_series[:, -1],
        ], dim=1)

        freq_features = torch.fft.rfft(case_series, dim=1).abs()[:, :self.freq_dim]

        if self.covariate_indices:
            cov_input = x[:, :, self.covariate_indices].reshape(batch_size, -1)
        else:
            cov_input = torch.zeros(batch_size, self.lookback, device=x.device).reshape(batch_size, -1)

        experts = torch.stack([
            self.temporal(flattened),
            self.trend(trend_features),
            self.frequency(freq_features),
            self.covariate(cov_input),
        ], dim=1)

        active_mask = torch.tensor([
            1.0,
            1.0 if self.use_trend and not self.temporal_only else 0.0,
            1.0 if self.use_frequency and not self.temporal_only else 0.0,
            1.0 if self.use_covariate and not self.temporal_only else 0.0,
        ], device=x.device)

        if self.temporal_only:
            weights = torch.zeros(batch_size, 4, device=x.device)
            weights[:, 0] = 1.0
        else:
            logits = self.gating(flattened)
            logits = logits.masked_fill(active_mask.unsqueeze(0) == 0, -1e9)
            weights = torch.softmax(logits, dim=-1)

        weights = weights.unsqueeze(-1)
        output = self.softplus((weights * experts).sum(dim=1))
        return output, weights.squeeze(-1), experts


class MixLinearTemporalLearner:
    """Standalone MixLinear temporal signal learner.

    This wraps _MixLinearNet with training, prediction, and MC-dropout
    uncertainty estimation. When used inside HantaCast, the predicted
    temporal signal modulates the SEIRD beta_t pathway.
    """

    def __init__(
        self,
        lookback: int = 3,
        horizon: int = 1,
        random_state: int = 42,
        epochs: int = 80,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-3,
        dropout: float = 0.1,
        mc_samples: int = 50,
        use_trend: bool = True,
        use_frequency: bool = True,
        use_covariate: bool = True,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.random_state = random_state
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.mc_samples = mc_samples
        self.use_trend = use_trend
        self.use_frequency = use_frequency
        self.use_covariate = use_covariate
        self.feature_columns: list[str] = []
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.model: Optional[_MixLinearNet] = None
        self.epochs_run: int = 0
        self.device = torch.device("cuda" if torch is not None and torch.cuda.is_available() else "cpu") if torch is not None else None
        self.train_df: Optional[pd.DataFrame] = None

    @staticmethod
    def _loss_fn(pred, target):
        mse = torch.mean((pred - target) ** 2)
        mae = torch.mean(torch.abs(pred - target))
        return 0.5 * mse + 0.5 * mae

    def fit(self, train_df: pd.DataFrame):
        set_global_seed(self.random_state)
        self.train_df = train_df.copy()

        self.feature_columns = select_feature_columns(train_df)
        dataset = make_supervised_windows(train_df, self.lookback, self.horizon, feature_columns=self.feature_columns)
        X = dataset["X"]
        y = dataset["y"]
        if len(X) < 1:
            raise RuntimeError("Insufficient supervised windows for MixLinear training.")

        self.mean_ = X.reshape(-1, X.shape[-1]).mean(axis=0)
        self.std_ = X.reshape(-1, X.shape[-1]).std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        Xn = (X - self.mean_) / self.std_

        n_samples = len(Xn)
        if n_samples == 1:
            X_train, X_val = Xn, Xn
            y_train, y_val = y, y
        else:
            split_idx = max(1, int(np.floor(n_samples * 0.8)))
            if split_idx >= n_samples:
                split_idx = n_samples - 1
            X_train, X_val = Xn[:split_idx], Xn[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            if len(X_val) == 0:
                X_train, X_val = Xn[:-1], Xn[-1:]
                y_train, y_val = y[:-1], y[-1:]

        covariate_indices = [idx for idx, col in enumerate(self.feature_columns) if col not in {"new_cases", "cumulative_cases"}]
        self.model = _MixLinearNet(
            self.lookback, len(self.feature_columns), self.horizon, covariate_indices,
            dropout=self.dropout, use_trend=self.use_trend, use_frequency=self.use_frequency,
            use_covariate=self.use_covariate, temporal_only=False,
        )
        self.model = self.model.to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
            batch_size=min(8, len(X_train)), shuffle=False,
        )
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        best_state = copy.deepcopy(self.model.state_dict())
        best_loss = float("inf")
        patience_counter = 0
        patience = 10

        for epoch in range(self.epochs):
            self.model.train()
            train_loss_sum = 0.0
            train_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred, _, _ = self.model(xb)
                loss = self._loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item())
                train_batches += 1
            self.model.eval()
            with torch.no_grad():
                val_pred, _, _ = self.model(X_val_t)
                val_loss = float(self._loss_fn(val_pred, y_val_t).item())
            train_loss_avg = train_loss_sum / max(train_batches, 1)
            log.info(f"  Epoch {epoch+1:3d}/{self.epochs} | train_loss={train_loss_avg:.4f} | val_loss={val_loss:.4f}")
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                log.info(f"  Early stopping at epoch {epoch+1}")
                break

        self.model.load_state_dict(best_state)
        self.model.eval()
        self.epochs_run = epoch + 1
        return self

    def predict_signal(self, future_covariates: Optional[pd.DataFrame] = None, horizon: int | None = None) -> np.ndarray:
        """Predict the MixLinear temporal signal (normalized to [0, 1])."""
        horizon = horizon or self.horizon
        x = self._prepare_last_window(future_covariates)
        xt = torch.tensor(x[None, :, :], dtype=torch.float32, device=self.device)
        self.model.train()  # enable dropout for MC sampling
        samples = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                sample, _, _ = self.model(xt)
                samples.append(sample.detach().cpu().numpy().reshape(-1))
        self.model.eval()
        draws = np.vstack(samples)
        mean_pred = draws.mean(axis=0)[:horizon]
        scale = max(float(np.nanmax(mean_pred)) if mean_pred.size else 0.0, 1.0)
        return np.clip(mean_pred / scale, 0.0, 1.0)

    def predict(self, future_covariates: Optional[pd.DataFrame] = None, horizon: int | None = None) -> np.ndarray:
        """Predict raw new cases (unnormalized)."""
        horizon = horizon or self.horizon
        x = self._prepare_last_window(future_covariates)
        xt = torch.tensor(x[None, :, :], dtype=torch.float32, device=self.device)
        samples = []
        self.model.train()
        with torch.no_grad():
            for _ in range(self.mc_samples):
                sample, _, _ = self.model(xt)
                samples.append(sample.detach().cpu().numpy().reshape(-1))
        self.model.eval()
        draws = np.vstack(samples)
        return draws.mean(axis=0)[:horizon]

    def _prepare_last_window(self, future_covariates: Optional[pd.DataFrame]) -> np.ndarray:
        assert self.train_df is not None and self.mean_ is not None and self.std_ is not None
        window = self.train_df[self.feature_columns].tail(self.lookback).copy()
        if len(window) < self.lookback:
            pad = pd.DataFrame([window.iloc[0]] * (self.lookback - len(window)), columns=window.columns)
            window = pd.concat([pad, window], ignore_index=True)
        if future_covariates is not None:
            replace_cols = [col for col in self.feature_columns if col in future_covariates.columns and col not in {"new_cases", "cumulative_cases"}]
            k = min(len(future_covariates), self.lookback)
            for col in replace_cols:
                window.iloc[-k:, window.columns.get_loc(col)] = future_covariates[col].to_numpy()[:k]
        array = window.to_numpy(dtype=float)
        return (array - self.mean_) / self.std_
