"""Causal analytical baseline implementing the presentation's complete algorithm path."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, DMatrix
from .synthetic import PARAMETERS, PARAMETER_INFO, generate, validate_measurements, stress_profile


def stress_features(frame):
    t = frame.temperature_c.to_numpy() - 125
    v = frame.voltage_v.to_numpy() - 4
    return np.column_stack([t, t*t, v, t*v])


def costs(probability, static_fail, cfn, cfp, warning=False, ready=True):
    accept_cost = float(probability * cfn)
    hold_cost = float((1 - probability) * cfp)
    action = "REJECT" if static_fail else "WARMUP" if not ready else "HOLD" if accept_cost >= hold_cost else "WATCH" if warning else "CONTINUE"
    return {"action": action, "accept_cost": accept_cost, "hold_cost": hold_cost,
            "threshold": cfp / (cfn + cfp), "static_override": bool(static_fail)}


class AnalyticalEngine:
    version = "phase1.1"

    def __init__(self, config):
        self.config = config
        baseline = generate(config, 36, "baseline", True)
        threshold = generate(config, 24, "threshold", True)
        risk_train = generate(config, 72, "risk_train")
        calibration = generate(config, 36, "risk_calibration")
        self.split_ids = {name: frame.component_id.unique().tolist() for name, frame in
                          [("baseline", baseline), ("threshold", threshold), ("risk_train", risk_train), ("risk_calibration", calibration)]}
        self.stress = LinearRegression().fit(stress_features(baseline), baseline[PARAMETERS])
        resid = baseline[PARAMETERS].to_numpy() - self.stress.predict(stress_features(baseline))
        self.scale = StandardScaler().fit(resid)
        z = self.scale.transform(resid)
        self.baseline_corr = np.corrcoef(z, rowvar=False)
        self.baseline_cov = np.cov(z, rowvar=False)
        self.isolation = IsolationForest(n_estimators=120, max_samples=256, contamination="auto", random_state=config.seed, n_jobs=1).fit(z)
        normal = self.features(threshold)
        valid = normal[normal.ready]
        self.if_threshold = float(valid.isolation_score.quantile(.99))
        self.corr_threshold = float(valid.correlation_drift.quantile(.99))
        self.cov_threshold = float(valid.covariance_drift.quantile(.99))
        self.risk_columns = ["isolation_score", "residual_z_max", "correlation_drift", "covariance_drift", "projected_z", "spec_margin"]
        train = self.features(risk_train)
        train = train[train.ready]
        self.risk = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=.08, subsample=.85,
                                  colsample_bytree=.9, random_state=config.seed, n_jobs=1, eval_metric="logloss")
        self.risk.fit(train[self.risk_columns], train.truth_active.astype(int))
        cal = self.features(calibration)
        cal = cal[cal.ready]
        margins = self.risk.predict(cal[self.risk_columns], output_margin=True).reshape(-1, 1)
        self.calibrator = LogisticRegression(C=10).fit(margins, cal.truth_active.astype(int))
        self.baseline_summary = {"healthy_components": 60, "risk_training_components": 72, "calibration_components": 36,
                                 "window_samples": config.window, "stress_coefficients": self.stress.coef_.tolist(),
                                 "stress_intercepts": self.stress.intercept_.tolist(), "residual_scale": self.scale.scale_.tolist(),
                                 "stress_features": ["temperature - 125", "(temperature - 125)^2", "voltage - 4", "temperature × voltage (centered)"],
                                 "if_threshold": self.if_threshold, "correlation_threshold": self.corr_threshold,
                                 "covariance_threshold": self.cov_threshold, "threshold_quantile": .99,
                                 "correlation": self.baseline_corr.tolist(), "covariance": self.baseline_cov.tolist()}

    def features(self, measurements):
        validate_measurements(measurements)
        frame = measurements.copy()
        expected = self.stress.predict(stress_features(frame))
        residual = frame[PARAMETERS].to_numpy() - expected
        z = self.scale.transform(residual)
        for j, parameter in enumerate(PARAMETERS):
            frame[f"expected_{parameter}"] = expected[:, j]
            frame[f"residual_{parameter}"] = residual[:, j]
            frame[f"z_{parameter}"] = z[:, j]
        frame["isolation_score"] = -self.isolation.score_samples(z)
        frame["residual_z_max"] = np.abs(z).max(axis=1)
        lower = np.array([p["lower"] for p in PARAMETER_INFO])
        upper = np.array([p["upper"] for p in PARAMETER_INFO])
        values = frame[PARAMETERS].to_numpy()
        frame["spec_margin"] = np.minimum((values-lower)/(upper-lower), (upper-values)/(upper-lower)).min(axis=1)
        frame["static_fail"] = frame.spec_margin < 0
        result = []
        w = self.config.window
        x = np.arange(w) - (w-1)/2
        for _, group in frame.groupby("component_id", sort=False):
            group = group.copy()
            gz = group[[f"z_{p}" for p in PARAMETERS]]
            means = gz.rolling(w, min_periods=w).mean()
            variances = gz.rolling(w, min_periods=w).var()
            corr_distance = np.zeros(len(group))
            cov_distance = np.zeros(len(group))
            for a in range(3):
                cov_distance += (variances.iloc[:, a].to_numpy() - self.baseline_cov[a, a])**2
                for b in range(a+1, 3):
                    corr = gz.iloc[:, a].rolling(w).corr(gz.iloc[:, b]).to_numpy()
                    cov = gz.iloc[:, a].rolling(w).cov(gz.iloc[:, b]).to_numpy()
                    corr_distance += 2 * (corr - self.baseline_corr[a, b])**2
                    cov_distance += 2 * (cov - self.baseline_cov[a, b])**2
            group["correlation_drift"] = np.sqrt(corr_distance)
            group["covariance_drift"] = np.sqrt(cov_distance) / np.linalg.norm(self.baseline_cov)
            slopes = gz.rolling(w).apply(lambda a: np.dot(x, a)/np.dot(x, x), raw=True)
            group["projected_z"] = (means + slopes * ((w-1)/2+self.config.horizon_h)).abs().max(axis=1)
            for j, p in enumerate(PARAMETERS):
                group[f"delta_{p}"] = group[p].diff()
                group[f"rolling_mean_{p}"] = group[p].rolling(w).mean()
                group[f"rolling_std_{p}"] = group[p].rolling(w).std()
                group[f"drift_rate_{p}"] = slopes.iloc[:, j].to_numpy() * self.scale.scale_[j]
            group["ready"] = np.arange(len(group)) >= w-1
            result.append(group)
        return pd.concat(result, ignore_index=True)

    def score(self, measurements):
        frame = self.features(measurements)
        frame["risk_probability"] = np.nan
        ready = frame.ready
        margin = self.risk.predict(frame.loc[ready, self.risk_columns], output_margin=True)
        frame.loc[ready, "risk_probability"] = self.calibrator.predict_proba(margin.reshape(-1,1))[:, 1]
        frame["anomaly_flag"] = frame.isolation_score > self.if_threshold
        frame["drift_flag"] = ready & ((frame.correlation_drift > self.corr_threshold) | (frame.covariance_drift > self.cov_threshold))
        frame["action"] = [costs(0 if not r.ready else r.risk_probability, r.static_fail,
                                 self.config.false_negative_cost, self.config.false_positive_cost,
                                 r.anomaly_flag or r.drift_flag, r.ready)["action"] for r in frame.itertuples()]
        return frame

    def forecast(self, group):
        """OLS sees only the latest trailing window; interval is conditional on the planned stress."""
        group = group.tail(self.config.window)
        if len(group) < self.config.window:
            return {"status": "insufficient_history", "required_samples": self.config.window, "parameters": []}
        now = float(group.hour.iloc[-1])
        h = np.arange(now + 1, now + self.config.horizon_h + 1)
        # Future stress schedule is known; scenario/ground-truth columns are never read.
        t, v = stress_profile(h, self.config, float(group.stress_extra_c.iloc[-1]))
        future = pd.DataFrame({"temperature_c": t, "voltage_v": v})
        expected = self.stress.predict(stress_features(future))
        predictions = []
        for j, p in enumerate(PARAMETERS):
            model = sm.OLS(group[f"residual_{p}"].to_numpy(), sm.add_constant(group.hour.to_numpy() - now)).fit()
            pred = model.get_prediction(sm.add_constant(h-now)).summary_frame(alpha=.05)
            mean = pred["mean"].to_numpy() + expected[:, j]
            lo = pred["obs_ci_lower"].to_numpy() + expected[:, j]
            hi = pred["obs_ci_upper"].to_numpy() + expected[:, j]
            limits = PARAMETER_INFO[j]
            crossing = (mean < limits["lower"]) | (mean > limits["upper"])
            possible = (lo < limits["lower"]) | (hi > limits["upper"])
            predictions.append({"parameter": p, "hours": h.tolist(), "mean": mean.tolist(), "lower": lo.tolist(), "upper": hi.tolist(),
                                "slope_per_h": float(model.params[1]), "r_squared": float(model.rsquared),
                                "mean_crossing_h": float(h[crossing][0]) if crossing.any() else None,
                                "possible_crossing_h": float(h[possible][0]) if possible.any() else None})
        return {"status": "ready", "interval": "95% OLS observation interval; conditional on scheduled stress and a linear residual trend",
                "parameters": predictions}

    def explain(self, row):
        if not bool(row.ready):
            return {"status": "insufficient_history", "contributions": []}
        X = row[self.risk_columns].astype(float).to_frame().T
        contribution = self.risk.get_booster().predict(DMatrix(X, feature_names=self.risk_columns), pred_contribs=True)[0]
        margin = float(contribution.sum())
        probability = float(self.calibrator.predict_proba(np.array([[margin]]))[0,1])
        return {"status": "ready", "method": "XGBoost native TreeSHAP", "units": "uncalibrated log-odds",
                "bias": float(contribution[-1]), "raw_margin": margin, "raw_probability": float(expit(margin)),
                "calibrated_probability": probability,
                "calibration_slope": float(self.calibrator.coef_[0,0]), "calibration_intercept": float(self.calibrator.intercept_[0]),
                "contributions": [{"feature": k, "value": float(row[k]), "contribution": float(v)}
                                  for k,v in zip(self.risk_columns, contribution[:-1])]}


def performance(frame, config):
    ready = frame[frame.ready]
    truth = ready.truth_active.astype(int)
    predicted = ready.action.isin(["HOLD", "REJECT"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0,1]).ravel()
    sensitivity = []
    for ratio in [1, 5, 10, 20, 50, 100]:
        pred = (ready.risk_probability >= 1/(1+ratio)) | ready.static_fail
        a,b,c,d = confusion_matrix(truth, pred, labels=[0,1]).ravel()
        sensitivity.append({"cost_ratio": ratio, "threshold": 1/(1+ratio), "fp": int(b), "fn": int(c),
                            "total_cost": int(b+ratio*c), "holds_or_rejects": int(pred.sum())})
    by_scenario = []
    for name, g in ready.groupby("scenario"):
        positive = g.action.isin(["HOLD", "REJECT"])
        by_scenario.append({"scenario": name, "samples": len(g), "active_truth": int(g.truth_active.sum()),
                            "flagged": int(positive.sum()), "false_positives": int((positive & ~g.truth_active).sum()),
                            "missed": int((~positive & g.truth_active).sum())})
    return {"scope": "Observed held-out synthetic component-hours after warmup; windows overlap",
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "samples": len(ready),
            "precision": float(precision_score(truth, predicted, zero_division=0)),
            "recall": float(recall_score(truth, predicted, zero_division=0)),
            "f1": float(f1_score(truth, predicted, zero_division=0)),
            "brier": float(brier_score_loss(truth, ready.risk_probability)),
            "realized_cost": float(fp*config.false_positive_cost+fn*config.false_negative_cost),
            "cost_sensitivity": sensitivity, "by_scenario": by_scenario}
