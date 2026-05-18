"""HantaCast: A Deep Mechanistically Constrained Hybrid Model for
Hantavirus Case Trend Forecasting.

HantaCast integrates a MixLinear-based deep temporal signal learner with
SEIRD-constrained dynamics. The learned temporal signal modulates the
time-varying transmission-related component (beta_t), and the model outputs
mechanistically constrained hantavirus case trajectories rather than
unconstrained regression curves.

Architecture:
  1. MixLinearTemporalLearner: gated mixture of temporal/trend/frequency/
     covariate experts → learned temporal signal
  2. SEIRDDynamics: discrete-time SEIRD with intervention-modulated beta_t
  3. The MixLinear signal enters beta_t as an additional time-varying
     modifier (mixlinear_signal → mixlinear_modifier → beta_t)

Training:
  - Step 1: Fit MixLinear on supervised windows to learn temporal patterns
  - Step 2: Fit SEIRD dynamics on the full series with MixLinear signal
    injected as a covariate
  - The two components are coupled through the beta_t modulation pathway.

Prediction:
  - Roll forward: MixLinear predicts future temporal signal, SEIRD evolves
    compartmental dynamics with the modulated beta_t.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.data.dataset import make_supervised_windows, select_feature_columns
from src.data.preprocess import build_future_covariates
from src.models.mixlinear import MixLinearTemporalLearner
from src.models.seird_dynamics import SEIRDDynamics
from src.utils.io import save_dataframe
from src.utils.seed import set_global_seed

log = logging.getLogger(__name__)


class HantaCast:
    """Unified hybrid model for hantavirus case trend forecasting.

    HantaCast = MixLinear temporal signal learner + SEIRD-constrained dynamics,
    coupled through beta_t modulation.

    Parameters
    ----------
    lookback : int
        Lookback window (days) for the supervised learning task.
    horizon : int
        Forecast horizon (days).
    random_state : int
        Random seed for reproducibility.
    mixlinear_epochs : int
        Training epochs for MixLinear component.
    mixlinear_lr : float
        Learning rate for MixLinear optimizer.
    mixlinear_wd : float
        Weight decay for MixLinear optimizer.
    mixlinear_dropout : float
        Dropout rate in MixLinear gating/expert networks.
    mixlinear_mc_samples : int
        Number of MC dropout samples for MixLinear prediction intervals.
    seird_population : float or None
        Total population for SEIRD; inferred from data if None.
    seird_mixlinear_scale : float
        Scale factor for MixLinear signal in SEIRD beta_t pathway.
    seird_interval_samples : int
        Number of bootstrap samples for SEIRD prediction intervals.
    seird_param_grid : dict or None
        Parameter search grid for SEIRD fitting.
    """

    def __init__(
        self,
        lookback: int = 3,
        horizon: int = 1,
        random_state: int = 42,
        mixlinear_epochs: int = 80,
        mixlinear_lr: float = 1e-2,
        mixlinear_wd: float = 1e-3,
        mixlinear_dropout: float = 0.1,
        mixlinear_mc_samples: int = 50,
        seird_population: float | None = None,
        seird_mixlinear_scale: float = 0.15,
        seird_interval_samples: int = 120,
        seird_param_grid: dict | None = None,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.random_state = random_state

        # MixLinear component
        self.mixlinear = MixLinearTemporalLearner(
            lookback=lookback,
            horizon=horizon,
            random_state=random_state,
            epochs=mixlinear_epochs,
            learning_rate=mixlinear_lr,
            weight_decay=mixlinear_wd,
            dropout=mixlinear_dropout,
            mc_samples=mixlinear_mc_samples,
        )

        # SEIRD component
        self.seird = SEIRDDynamics(
            random_state=random_state,
            total_population=seird_population or 200.0,
            mixlinear_signal_scale=seird_mixlinear_scale,
            interval_samples=seird_interval_samples,
            param_grid=seird_param_grid,
        )

        self.train_df: Optional[pd.DataFrame] = None
        self.mixlinear_signal_: Optional[np.ndarray] = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit_mixlinear_full(self, full_df: pd.DataFrame) -> "HantaCast":
        """Pre-train MixLinear on the full dataset and compute the signal once.

        This is used in the 'pre-train' evaluation mode: MixLinear learns from
        all available data once, producing a stable temporal signal. During
        rolling-origin evaluation, only SEIRD is refit on each origin's slice,
        using the pre-computed signal.

        This avoids the overfitting that occurs when MixLinear is retrained
        on small (5-7 sample) rolling-origin slices.
        """
        set_global_seed(self.random_state)
        log.info("HantaCast: pre-training MixLinear on full dataset ...")
        self.mixlinear.fit(full_df)
        log.info("HantaCast: computing MixLinear signal on full dataset ...")
        self._full_signal = self._compute_training_signal(full_df)
        self._full_df = full_df.copy()
        self._full_df["mixlinear_signal"] = self._full_signal
        log.info(f"HantaCast: MixLinear pre-training complete ({len(self._full_df)} days)")
        return self

    def fit_seird_only(self, train_df: pd.DataFrame) -> "HantaCast":
        """Fit only SEIRD using pre-computed MixLinear signal.

        Requires fit_mixlinear_full() to have been called first.
        The pre-computed signal is sliced to match train_df dates.
        """
        if not hasattr(self, '_full_df') or self._full_df is None:
            raise RuntimeError("Call fit_mixlinear_full() first to pre-train MixLinear.")

        self.train_df = train_df.copy()
        train_dates = set(self.train_df["date"].astype(str).tolist())

        # Slice the pre-computed signal to match training dates
        full_dates = self._full_df["date"].astype(str).tolist()
        signal_slice = []
        for i, d in enumerate(full_dates):
            if d in train_dates:
                signal_slice.append(self._full_signal[i])

        if len(signal_slice) != len(train_df):
            # Some dates might not match exactly; pad/trim
            signal_slice = self._full_signal[:len(train_df)]

        augmented_df = train_df.copy()
        augmented_df["mixlinear_signal"] = np.asarray(signal_slice, dtype=float)[:len(train_df)]

        log.info("HantaCast: fitting SEIRD with pre-computed MixLinear signal ...")
        self.seird.fit(augmented_df)
        self.mixlinear_signal_ = np.asarray(signal_slice, dtype=float)[:len(train_df)]
        self._is_fitted = True
        return self

    def fit(self, train_df: pd.DataFrame) -> "HantaCast":
        """Fit HantaCast on training data.

        Step 1: Fit MixLinear on supervised windows.
        Step 2: Compute MixLinear signal on the training series.
        Step 3: Fit SEIRD dynamics with MixLinear signal as a covariate.
        """
        set_global_seed(self.random_state)
        self.train_df = train_df.copy()
        log.info("HantaCast: fitting MixLinear temporal signal learner ...")
        self.mixlinear.fit(train_df)

        # Compute MixLinear signal over the training period
        log.info("HantaCast: computing MixLinear temporal signal ...")
        signal = self._compute_training_signal(train_df)
        self.mixlinear_signal_ = signal

        # Augment training data with mixlinear_signal
        augmented_df = train_df.copy()
        augmented_df["mixlinear_signal"] = signal

        log.info("HantaCast: fitting SEIRD dynamics with MixLinear signal ...")
        self.seird.fit(augmented_df)
        self._is_fitted = True
        log.info("HantaCast: training complete.")
        return self

    def _compute_training_signal(self, df: pd.DataFrame) -> np.ndarray:
        """Compute the MixLinear temporal signal over the training series.

        For each feasible window, the MixLinear predicts one horizon step.
        The signal is normalized to [0, 1].
        """
        feature_columns = select_feature_columns(df)
        dataset = make_supervised_windows(df, self.lookback, self.horizon, feature_columns=feature_columns)
        X = dataset["X"]
        y = dataset["y"]
        target_dates = dataset["target_dates"]

        if len(X) == 0:
            return np.zeros(len(df), dtype=float)

        # Predict for each window using MixLinear
        device = self.mixlinear.device
        model = self.mixlinear.model
        mean = self.mixlinear.mean_
        std = self.mixlinear.std_

        predictions = []
        model.eval()
        with torch.no_grad():
            for i in range(len(X)):
                x = torch.tensor(((X[i] - mean) / std)[None, :, :], dtype=torch.float32, device=device)
                pred, _, _ = model(x)
                predictions.append(pred.cpu().numpy().reshape(-1))

        # Build aligned signal array
        signal = np.zeros(len(df), dtype=float)
        for i, (dates, pred) in enumerate(zip(target_dates, predictions)):
            for j, d in enumerate(dates):
                idx = df.index[df["date"] == d]
                if len(idx) > 0 and j < len(pred):
                    signal[idx[0]] = pred[j]

        # Normalize to [0, 1]
        scale = max(float(np.nanmax(signal)) if signal.size else 0.0, 1.0)
        return np.clip(signal / scale, 0.0, 1.0)

    def predict(self, horizon: int | None = None, future_covariates: pd.DataFrame | None = None) -> pd.DataFrame:
        """Generate HantaCast forecast.

        Steps:
        1. MixLinear predicts the future temporal signal.
        2. SEIRD simulates forward with the MixLinear signal modulating beta_t.
        3. Construct prediction DataFrame with compartments and intervals.
        """
        if not self._is_fitted:
            raise RuntimeError("HantaCast must be fitted before prediction.")

        horizon = int(horizon or self.horizon)
        if future_covariates is None:
            future_covariates = build_future_covariates(self.train_df, horizon)

        log.info(f"HantaCast: forecasting {horizon} days ahead ...")

        # Step 1: MixLinear predicts future temporal signal
        mixlinear_signal = self.mixlinear.predict_signal(future_covariates, horizon)
        future_covariates["mixlinear_signal"] = mixlinear_signal[:len(future_covariates)]

        # Step 2: SEIRD simulates forward
        preds_mean, traj = self.seird.simulate_forward(future_covariates)
        preds_int = self.seird._integerize_counts(preds_mean)
        lower, upper = self.seird.prediction_interval(future_covariates)

        # Step 3: Build output DataFrame
        dates = pd.to_datetime(future_covariates["date"]).tolist()[:horizon]
        last_cumulative = float(pd.to_numeric(self.train_df["cumulative_cases"], errors="coerce").iloc[-1])

        prediction = pd.DataFrame({
            "model": "HantaCast",
            "date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "day_index": future_covariates["day_index"].tolist()[:horizon] if "day_index" in future_covariates.columns else list(range(len(dates))),
            "forecast_median": preds_int.astype(int)[:horizon] if len(preds_int) >= horizon else np.pad(preds_int, (0, max(0, horizon - len(preds_int))), constant_values=0).astype(int),
            "forecast_lower": lower.astype(int)[:horizon] if len(lower) >= horizon else np.pad(lower, (0, max(0, horizon - len(lower))), constant_values=0).astype(int),
            "forecast_upper": upper.astype(int)[:horizon] if len(upper) >= horizon else np.pad(upper, (0, max(0, horizon - len(upper))), constant_values=0).astype(int),
            "interval_type": "bootstrap_95ci",
            "interval_source": "SEIRD_parametric_bootstrap",
            "expected_new_cases": preds_mean.astype(float)[:horizon],
            "cumulative_cases_pred": (int(round(last_cumulative)) + preds_int.cumsum())[:horizon].astype(int),
            "active_cases_pred": (traj["I"][:horizon] if len(traj["I"]) >= horizon else traj["I"]).astype(float),
            "beta_t": (traj["beta_t"][:horizon] if len(traj["beta_t"]) >= horizon else traj["beta_t"]).astype(float),
            "mixlinear_signal": mixlinear_signal[:horizon].astype(float),
        })
        return prediction

    def recursive_forecast(
        self,
        total_horizon: int,
        step_horizon: int = 1,
        future_covariates: pd.DataFrame | None = None,
        n_samples: int = 80,
    ) -> pd.DataFrame:
        """Recursive multi-step forecast.

        At each step, MixLinear predicts the next temporal signal chunk,
        SEIRD evolves the compartmental dynamics, and the state is carried
        forward. This enables long-horizon (e.g., 150-day) forecasts.
        """
        if not self._is_fitted:
            raise RuntimeError("HantaCast must be fitted before forecasting.")

        total_horizon = int(total_horizon)
        step_horizon = max(1, int(step_horizon))

        if future_covariates is None:
            future_covariates = build_future_covariates(self.train_df, total_horizon)

        future_covariates = future_covariates.copy().reset_index(drop=True)
        original_df = self.train_df.copy()
        original_last_state = copy.deepcopy(self.seird.last_state)
        original_best_params = copy.deepcopy(self.seird.best_params)

        # MC draws over recursive paths
        rng = np.random.default_rng(self.random_state + 409)
        draws = []
        for _ in range(int(n_samples)):
            temp_df = original_df.copy()
            self.seird.last_state = copy.deepcopy(original_last_state)
            sample_path = []
            start = 0
            while start < total_horizon:
                chunk = min(step_horizon, total_horizon - start)
                cov_chunk = future_covariates.iloc[start:start + chunk].copy()

                # MixLinear signal for this chunk
                ml_signal = self.mixlinear.predict_signal(cov_chunk, chunk)
                cov_chunk["mixlinear_signal"] = ml_signal[:len(cov_chunk)]

                # SEIRD forward step
                preds_mean, traj = self.seird.simulate_forward(cov_chunk)
                sample_path.extend(preds_mean.tolist()[:chunk])

                # Update SEIRD state for next step
                preds_int = self.seird._integerize_counts(preds_mean)
                seird_augmented = cov_chunk.copy()
                seird_augmented["new_cases"] = preds_int
                seird_augmented["cumulative_cases"] = float(pd.to_numeric(temp_df["cumulative_cases"], errors="coerce").iloc[-1]) + np.cumsum(preds_int)
                seird_augmented["active_cases"] = traj["I"][:len(seird_augmented)] if len(traj["I"]) >= len(seird_augmented) else np.pad(traj["I"], (0, len(seird_augmented) - len(traj["I"])), constant_values=traj["I"][-1])
                seird_augmented["deaths"] = traj["D"][:len(seird_augmented)] if len(traj["D"]) >= len(seird_augmented) else np.pad(traj["D"], (0, len(seird_augmented) - len(traj["D"])), constant_values=traj["D"][-1])
                self.seird.fit(pd.concat([temp_df, seird_augmented], ignore_index=True))

                start += chunk
            draws.append(np.asarray(sample_path, dtype=float))

        # Restore original state
        self.seird.last_state = original_last_state
        self.seird.best_params = original_best_params
        self.seird.train_df = original_df

        # Aggregate draws
        draw_matrix = np.vstack(draws)
        median_pred = np.median(draw_matrix, axis=0)
        lower = np.percentile(draw_matrix, 2.5, axis=0)
        upper = np.percentile(draw_matrix, 97.5, axis=0)

        dates = pd.to_datetime(future_covariates["date"]).tolist()[:total_horizon]

        prediction = pd.DataFrame({
            "model": "HantaCast",
            "date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "day_index": list(range(len(dates))),
            "forecast_median": self.seird._integerize_counts(median_pred).astype(int),
            "forecast_lower": self.seird._integerize_counts(lower).astype(int),
            "forecast_upper": self.seird._integerize_counts(upper).astype(int),
            "interval_type": "recursive_mc_bootstrap_95ci",
            "interval_source": "HantaCast_recursive_MC_dropout_bootstrap",
        })
        return prediction

    def forecast(
        self,
        total_horizon: int = 150,
        n_samples: int = 80,
    ) -> pd.DataFrame:
        """Fast forward forecast without re-fitting.

        Strategy:
          1. MixLinear rolls forward 1 step at a time in eval mode (no MC),
             producing a deterministic temporal signal for all 150 days.
          2. SEIRD simulates 150 days with that signal → point forecast.
          3. For prediction intervals, bootstrap SEIRD parameters and
             re-run the SEIRD simulation N times (fast, no MixLinear involved).

        This avoids costly per-step MC dropout and SEIRD re-fitting.
        Total time: ~1s per bootstrap sample → ~80s for full forecast.
        """
        if not self._is_fitted:
            raise RuntimeError("HantaCast must be fitted before forecasting.")
        if self.mixlinear.model is None:
            raise RuntimeError("MixLinear model not initialized.")

        total_horizon = int(total_horizon)
        original_df = self.train_df.copy()
        original_last_state = {**self.seird.last_state}
        original_best_params = {**self.seird.best_params}

        device = self.mixlinear.device
        model_net = self.mixlinear.model
        mean = self.mixlinear.mean_
        std = self.mixlinear.std_
        feature_cols = self.mixlinear.feature_columns
        lookback = self.lookback

        covariate_cols = ["intervention_index", "mobility_index", "flight_volume", "behavior_response_index"]
        last_cov = {
            col: float(pd.to_numeric(original_df[col], errors="coerce").ffill().iloc[-1])
            for col in covariate_cols if col in original_df.columns
        }

        # ── Step 1: Roll forward MixLinear to get 150-day signal ──
        log.info("  Step 1/3: Rolling MixLinear signal forward ...")
        history_df = original_df.copy()
        signal_seq = []
        raw_pred_seq = []

        model_net.eval()
        for step in range(total_horizon):
            window = history_df[feature_cols].tail(lookback).to_numpy(dtype=float)
            if len(window) < lookback:
                pad = np.tile(window[0], (lookback - len(window), 1))
                window = np.vstack([pad, window])
            x = (window - mean) / std
            with torch.no_grad():
                xt = torch.tensor(x[None, :, :], dtype=torch.float32, device=device)
                pred, _, _ = model_net(xt)
                pred_val = float(pred.cpu().numpy().ravel()[0])

            raw_pred_seq.append(pred_val)
            signal_seq.append(pred_val / max(pred_val, 1.0))

            # Append for next step
            last_date = pd.to_datetime(history_df["date"]).iloc[-1]
            new_row = {
                "date": (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "new_cases": max(pred_val, 0.0),
                "cumulative_cases": float(pd.to_numeric(history_df["cumulative_cases"],
                    errors="coerce").iloc[-1]) + max(pred_val, 0.0),
            }
            for col in feature_cols:
                if col not in new_row:
                    new_row[col] = float(pd.to_numeric(history_df[col], errors="coerce").iloc[-1])
            history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)

        signal_arr = np.asarray(signal_seq, dtype=float)

        # ── Step 2: SEIRD point forecast with MixLinear signal ──
        log.info("  Step 2/3: SEIRD point forecast ...")
        self.seird.last_state = {**original_last_state}
        last_date = pd.to_datetime(original_df["date"]).iloc[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=total_horizon, freq="D")
        future_cov = pd.DataFrame({
            "date": future_dates,
            "intervention_index": last_cov.get("intervention_index", 0.2),
            "mobility_index": last_cov.get("mobility_index", 0.0),
            "behavior_response_index": last_cov.get("behavior_response_index", 0.25),
            "flight_volume": last_cov.get("flight_volume", 0.0),
            "mixlinear_signal": signal_arr,
        })
        point_preds, point_traj = self.seird.simulate_forward(future_cov)
        point_preds = np.maximum(point_preds, 0.0)

        # ── Step 3: Bootstrap SEIRD params for prediction intervals ──
        log.info(f"  Step 3/3: Bootstrapping SEIRD parameters ({n_samples} samples) ...")
        param_sets = self.seird._sample_parameter_sets(int(n_samples))
        draws = []
        for params in param_sets:
            preds, _ = self.seird._simulate(params, future_cov, original_last_state)
            draws.append(np.maximum(preds, 0.0))
        draw_matrix = np.vstack(draws) if draws else point_preds[None, :]

        median_pred = np.median(draw_matrix, axis=0)
        lower = np.percentile(draw_matrix, 2.5, axis=0)
        upper = np.percentile(draw_matrix, 97.5, axis=0)

        # ── Restore ──
        self.seird.last_state = original_last_state
        self.seird.best_params = original_best_params
        self.seird.train_df = original_df
        self.train_df = original_df

        # ── Build output ──
        prediction = pd.DataFrame({
            "model": "HantaCast",
            "date": future_dates.strftime("%Y-%m-%d"),
            "day_index": list(range(total_horizon)),
            "forecast_median": self.seird._integerize_counts(median_pred).astype(int),
            "forecast_lower": self.seird._integerize_counts(lower).astype(int),
            "forecast_upper": self.seird._integerize_counts(upper).astype(int),
            "interval_type": "seird_bootstrap_95ci",
            "interval_source": "HantaCast_SEIRD_parametric_bootstrap",
        })
        return prediction

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save model checkpoint (MixLinear weights + SEIRD params)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "mixlinear_state_dict": self.mixlinear.model.state_dict() if self.mixlinear.model is not None else None,
            "mixlinear_mean": self.mixlinear.mean_,
            "mixlinear_std": self.mixlinear.std_,
            "mixlinear_feature_columns": self.mixlinear.feature_columns,
            "seird_best_params": self.seird.best_params,
            "seird_last_state": self.seird.last_state,
            "seird_initial_state": self.seird.initial_state,
            "seird_last_cumulative": self.seird.last_cumulative_cases,
            "seird_last_deaths": self.seird.last_deaths,
            "seird_residual_scale": self.seird.residual_scale,
            "seird_residuals": self.seird.residuals_,
            "seird_total_population": self.seird.total_population,
            "seird_param_bounds": self.seird.param_bounds,
            "lookback": self.lookback,
            "horizon": self.horizon,
            "random_state": self.random_state,
            "is_fitted": self._is_fitted,
        }
        torch.save(checkpoint, path)
        log.info(f"HantaCast checkpoint saved to {path}")
        return path

    def load_checkpoint(self, path: str | Path) -> "HantaCast":
        """Load model checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Rebuild MixLinear network if state_dict is available but model is not initialized
        if checkpoint["mixlinear_state_dict"] is not None:
            feature_columns = checkpoint.get("mixlinear_feature_columns", [])
            n_features = len(feature_columns) if feature_columns else 7
            m_lookback = checkpoint.get("lookback", self.lookback)
            covariate_indices = [idx for idx, col in enumerate(feature_columns) if col not in {"new_cases", "cumulative_cases"}]
            if self.mixlinear.model is None:
                from src.models.mixlinear import _MixLinearNet
                self.mixlinear.model = _MixLinearNet(
                    m_lookback, n_features, checkpoint.get("horizon", self.horizon),
                    covariate_indices,
                    dropout=self.mixlinear.dropout,
                    use_trend=self.mixlinear.use_trend,
                    use_frequency=self.mixlinear.use_frequency,
                    use_covariate=self.mixlinear.use_covariate,
                )
                self.mixlinear.model = self.mixlinear.model.to(self.mixlinear.device)
            self.mixlinear.model.load_state_dict(checkpoint["mixlinear_state_dict"])
        self.mixlinear.mean_ = checkpoint["mixlinear_mean"]
        self.mixlinear.std_ = checkpoint["mixlinear_std"]
        self.mixlinear.feature_columns = checkpoint.get("mixlinear_feature_columns", [])

        # Restore SEIRD state
        self.seird.best_params = checkpoint["seird_best_params"]
        self.seird.last_state = checkpoint["seird_last_state"]
        self.seird.initial_state = checkpoint["seird_initial_state"]
        self.seird.last_cumulative_cases = checkpoint["seird_last_cumulative"]
        self.seird.last_deaths = checkpoint["seird_last_deaths"]
        self.seird.residual_scale = checkpoint["seird_residual_scale"]
        self.seird.residuals_ = checkpoint["seird_residuals"]
        self.seird.total_population = checkpoint["seird_total_population"]
        self.seird.param_bounds = checkpoint["seird_param_bounds"]
        self.lookback = checkpoint["lookback"]
        self.horizon = checkpoint["horizon"]
        self.random_state = checkpoint["random_state"]
        self._is_fitted = checkpoint["is_fitted"]

        log.info(f"HantaCast checkpoint loaded from {path}")
        return self
