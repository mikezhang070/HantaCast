from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


class SEIRDDynamics:
    """Discrete-time SEIRD epidemiological dynamics with intervention-modulated
    transmission rate (beta_t).

    The transmission rate is modulated by:
      - beta0: baseline transmission rate
      - intervention_index: non-pharmaceutical intervention intensity
      - mobility_index: aggregate mobility indicator
      - behavior_response_index: aggregate behavioral response
      - flight_volume: repatriation flight volume
      - mixlinear_signal: learned temporal signal from MixLinear (HantaCast integration)
      - Dynamic parameters: alpha, rho, eta, phi, q

    All outputs are aggregate-level compartment counts.
    """

    def __init__(
        self,
        random_state: int = 42,
        total_population: float = 200.0,
        mixlinear_signal_scale: float = 0.15,
        interval_samples: int = 120,
        param_grid: dict | None = None,
    ):
        self.random_state = random_state
        self.total_population = total_population
        self.mixlinear_signal_scale = mixlinear_signal_scale
        self.interval_samples = interval_samples
        self.best_params: dict[str, float] = {}
        self.initial_state: dict[str, float] = {}
        self.last_state: dict[str, float] = {}
        self.last_cumulative_cases: float = 0.0
        self.last_deaths: float = 0.0
        self.residual_scale: float = 1.0
        self.residuals_: np.ndarray = np.asarray([0.0], dtype=float)
        self.param_bounds: dict[str, tuple[float, float]] = {}
        self.train_df: Optional[pd.DataFrame] = None  # noqa: UP007

        self._default_param_grid = {
            "beta0": [0.35, 0.70],
            "sigma": [0.14, 0.20],
            "gamma": [0.10, 0.18],
            "mu": [0.0, 0.02],
            "alpha": [0.2, 0.5],
            "rho": [0.0, 0.15],
            "eta": [0.0, 0.15],
            "phi": [0.0, 0.05],
            "q": [0.0, 0.2],
            "reporting_rate": [0.85, 1.0],
            "flight_scale": [5.0],
        }
        self._param_grid = param_grid or self._default_param_grid

    @staticmethod
    def _integerize_counts(values: np.ndarray) -> np.ndarray:
        return np.maximum(np.rint(np.asarray(values, dtype=float)), 0.0).astype(int)

    @staticmethod
    def _prepare_covariates(df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        defaults = {
            "intervention_index": 0.2, "mobility_index": 0.0,
            "behavior_response_index": 0.25, "flight_volume": 0.0,
            "mixlinear_signal": 0.0,
        }
        for col, default in defaults.items():
            if col not in work.columns:
                work[col] = default
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(default)
        return work[list(defaults.keys())]

    def _infer_initial_state(self, work: pd.DataFrame) -> dict[str, float]:
        cumulative = pd.to_numeric(work["cumulative_cases"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
        active = pd.to_numeric(work["active_cases"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
        new_cases = pd.to_numeric(work["new_cases"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        deaths = pd.to_numeric(work.get("deaths", pd.Series(0.0)), errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)

        cumulative_max = float(np.nanmax(cumulative)) if len(cumulative) else 0.0
        active_max = float(np.nanmax(active)) if len(active) else 0.0
        deaths_max = float(np.nanmax(deaths)) if len(deaths) else 0.0
        population_hint = max(200.0, 12.0 * cumulative_max + 8.0 * active_max + 25.0 * deaths_max)

        i0 = float(max(active[0], new_cases[0], 1.0))
        e0 = float(max(new_cases[: min(3, len(new_cases))].mean() * 1.5, 1.0))
        d0 = float(max(deaths[0], 0.0))
        r0 = float(max(cumulative[0] - i0 - d0, 0.0))
        s0 = float(max(population_hint - e0 - i0 - r0 - d0, 1.0))
        return {"S": s0, "E": e0, "I": i0, "R": r0, "D": d0, "N": population_hint}

    def _simulate(self, params: dict[str, float], covariates: pd.DataFrame, state0: dict[str, float]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        s = float(state0["S"])
        e = float(state0["E"])
        i = float(state0["I"])
        r = float(state0["R"])
        d = float(state0["D"])
        n = float(state0["N"])

        report_mean = []
        s_hist, e_hist, i_hist, r_hist, d_hist = [], [], [], [], []
        beta_hist, lambda_hist = [], []

        for _, row in covariates.iterrows():
            intervention_index = float(row.get("intervention_index", 0.2))
            mobility_index = float(row.get("mobility_index", 0.0))
            behavior_response_index = float(row.get("behavior_response_index", 0.25))
            flight_volume = float(row.get("flight_volume", 0.0))
            mixlinear_signal = float(row.get("mixlinear_signal", 0.0))

            flight_modifier = 1.0 + params["phi"] * np.tanh(flight_volume / params["flight_scale"])
            intervention_modifier = max(0.05, 1.0 - params["alpha"] * intervention_index)
            mobility_modifier = max(0.05, 1.0 + params["rho"] * mobility_index)
            behavior_modifier = max(0.05, 1.0 + params["eta"] * behavior_response_index)
            mixlinear_modifier = max(0.05, 1.0 + self.mixlinear_signal_scale * mixlinear_signal)

            beta_t = params["beta0"] * intervention_modifier * mobility_modifier * behavior_modifier * flight_modifier * mixlinear_modifier
            beta_t = float(max(beta_t, 0.0))

            infectious_pool = i + params["q"] * e
            new_exposed = beta_t * s * infectious_pool / max(n, 1.0)
            new_infectious = params["sigma"] * e
            new_recovered = params["gamma"] * i
            new_deaths = params["mu"] * i
            observed_new_cases_mean = params["reporting_rate"] * params["sigma"] * e

            s = max(s - new_exposed, 0.0)
            e = max(e + new_exposed - new_infectious, 0.0)
            i = max(i + new_infectious - new_recovered - new_deaths, 0.0)
            r = max(r + new_recovered, 0.0)
            d = max(d + new_deaths, 0.0)

            total = s + e + i + r + d
            if total > 0 and abs(total - n) > 1e-6:
                scale = n / total
                s *= scale; e *= scale; i *= scale; r *= scale; d *= scale

            report_mean.append(max(observed_new_cases_mean, 0.0))
            s_hist.append(s); e_hist.append(e); i_hist.append(i)
            r_hist.append(r); d_hist.append(d)
            beta_hist.append(beta_t); lambda_hist.append(max(observed_new_cases_mean, 0.0))

        trajectories = {
            "S": np.asarray(s_hist, dtype=float), "E": np.asarray(e_hist, dtype=float),
            "I": np.asarray(i_hist, dtype=float), "R": np.asarray(r_hist, dtype=float),
            "D": np.asarray(d_hist, dtype=float), "beta_t": np.asarray(beta_hist, dtype=float),
            "lambda_t": np.asarray(lambda_hist, dtype=float),
        }
        return np.asarray(report_mean, dtype=float), trajectories

    def fit(self, train_df: pd.DataFrame):
        self.train_df = train_df.copy()
        work = train_df.copy()
        for col in ["new_cases", "cumulative_cases", "active_cases", "deaths"]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

        work["new_cases"] = work.get("new_cases", pd.Series(0.0, index=work.index)).fillna(0.0)
        if "cumulative_cases" not in work.columns:
            work["cumulative_cases"] = work["new_cases"].cumsum()
        work["cumulative_cases"] = pd.to_numeric(work["cumulative_cases"], errors="coerce").ffill().fillna(0.0)
        if "active_cases" not in work.columns:
            work["active_cases"] = work["new_cases"].rolling(min(7, max(3, len(work))), min_periods=1).sum()
        work["active_cases"] = pd.to_numeric(work["active_cases"], errors="coerce").ffill().fillna(work["new_cases"].rolling(3, min_periods=1).sum())
        if "deaths" not in work.columns:
            work["deaths"] = 0.0
        work["deaths"] = pd.to_numeric(work["deaths"], errors="coerce").ffill().fillna(0.0)

        observed = self._integerize_counts(work["new_cases"].to_numpy(dtype=float)).astype(float)
        observed_active = work["active_cases"].to_numpy(dtype=float)
        covariates = self._prepare_covariates(work)
        self.initial_state = self._infer_initial_state(work)
        self.total_population = float(self.initial_state["N"])

        grid = self._param_grid
        self.param_bounds = {key: (float(min(values)), float(max(values))) for key, values in grid.items()}

        best_score = float("inf")
        best_preds = np.zeros_like(observed)
        best_traj: dict[str, np.ndarray] = {}
        for values in product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            preds_mean, traj = self._simulate(params, covariates, self.initial_state)
            case_mae = float(np.mean(np.abs(preds_mean - observed)))
            case_rmse = float(np.sqrt(np.mean((preds_mean - observed) ** 2)))
            active_mae = float(np.mean(np.abs(traj["I"] - observed_active)))
            score = case_mae + case_rmse + 0.35 * active_mae
            if score < best_score:
                best_score = score
                self.best_params = params
                best_preds = preds_mean
                best_traj = traj

        residuals = observed - best_preds
        self.residuals_ = residuals.astype(float)
        self.residual_scale = float(np.std(residuals)) if residuals.size > 1 else max(float(np.std(residuals)), 1.0)
        self.last_cumulative_cases = float(work["cumulative_cases"].iloc[-1])
        self.last_deaths = float(work["deaths"].iloc[-1])
        self.last_state = {
            "S": float(best_traj["S"][-1]) if len(best_traj.get("S", [])) else float(self.initial_state["S"]),
            "E": float(best_traj["E"][-1]) if len(best_traj.get("E", [])) else float(self.initial_state["E"]),
            "I": float(best_traj["I"][-1]) if len(best_traj.get("I", [])) else float(self.initial_state["I"]),
            "R": float(best_traj["R"][-1]) if len(best_traj.get("R", [])) else float(self.initial_state["R"]),
            "D": float(best_traj["D"][-1]) if len(best_traj.get("D", [])) else float(self.initial_state["D"]),
            "N": float(self.total_population),
        }
        return self

    def simulate_forward(self, covariates: pd.DataFrame, params: dict[str, float] | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Simulate SEIRD forward given covariates and parameters."""
        p = params or self.best_params
        if not p:
            raise RuntimeError("SEIRD parameters not fitted. Call fit() first.")
        cov = self._prepare_covariates(covariates)
        return self._simulate(p, cov, self.last_state)

    def _sample_parameter_sets(self, n_samples: int) -> list[dict[str, float]]:
        if not self.best_params:
            return []
        rng = np.random.default_rng(self.random_state + 137)
        perturbation_scale = {
            "beta0": 0.12, "sigma": 0.10, "gamma": 0.10, "mu": 0.20,
            "alpha": 0.18, "rho": 0.18, "eta": 0.18, "phi": 0.20,
            "q": 0.20, "reporting_rate": 0.05, "flight_scale": 0.0,
        }
        samples = []
        for _ in range(int(n_samples)):
            draw = {}
            for key, value in self.best_params.items():
                low, high = self.param_bounds.get(key, (value, value))
                scale = perturbation_scale.get(key, 0.1)
                if scale <= 0 or low == high:
                    draw[key] = float(np.clip(value, low, high))
                    continue
                proposal = float(rng.normal(loc=value, scale=max(abs(value) * scale, 1.0e-4)))
                draw[key] = float(np.clip(proposal, low, high))
            samples.append(draw)
        return samples

    def prediction_interval(self, covariates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Bootstrap prediction interval for new cases."""
        horizon = len(covariates)
        if horizon == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        param_sets = self._sample_parameter_sets(self.interval_samples)
        if not param_sets:
            point_preds, _ = self._simulate(self.best_params, self._prepare_covariates(covariates), self.last_state)
            lower = self._integerize_counts(np.maximum(point_preds - 1.96 * self.residual_scale, 0.0))
            upper = self._integerize_counts(point_preds + 1.96 * self.residual_scale)
            return lower, upper
        rng = np.random.default_rng(self.random_state + 409)
        residuals = np.asarray(self.residuals_, dtype=float)
        if residuals.size == 0:
            residuals = np.asarray([0.0], dtype=float)
        draws = []
        horizon_scale = np.sqrt(1.0 + 0.03 * np.arange(1, horizon + 1, dtype=float))
        cov = self._prepare_covariates(covariates)
        for params in param_sets:
            preds_mean, _ = self._simulate(params, cov, self.last_state)
            sampled_noise = rng.choice(residuals, size=horizon, replace=True) * horizon_scale
            lambda_draw = np.maximum(preds_mean + sampled_noise, 0.0)
            draws.append(rng.poisson(lambda_draw))
        draw_matrix = np.vstack(draws)
        lower = self._integerize_counts(np.percentile(draw_matrix, 2.5, axis=0))
        upper = self._integerize_counts(np.percentile(draw_matrix, 97.5, axis=0))
        return lower, upper
