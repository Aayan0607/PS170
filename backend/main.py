import io
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .synthetic import SessionConfig, PARAMETERS, PARAMETER_INFO, SCENARIOS, generate
from .engine import AnalyticalEngine, costs, performance

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("NIRIKSHAN_DB", str(ROOT / "data" / "sessions.sqlite3")))
app = FastAPI(title="Nirikshan Netra", version="0.1.0", description="Synthetic burn-in screening prototype. Phase 1.")
cache = {}
lock = threading.Lock()
plot_lock = threading.Lock()


def db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created TEXT, config TEXT, measurements TEXT)")
    return connection


def records(frame):
    return json.loads(frame.to_json(orient="records"))


def load(session_id):
    # Small local demo cache; SQLite preserves source measurements across restarts.
    with lock:
        if session_id in cache:
            return cache[session_id]
        with db() as connection:
            row = connection.execute("SELECT config, measurements FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Session not found")
        config = SessionConfig.model_validate_json(row[0])
        raw = pd.read_json(io.StringIO(row[1]), orient="records")
        engine = AnalyticalEngine(config)
        frame = engine.score(raw)
        if len(cache) >= 3:
            cache.pop(next(iter(cache)))
        cache[session_id] = (config, engine, frame)
        return cache[session_id]


def observed(session_id, hour):
    config, engine, frame = load(session_id)
    hour = config.duration_h if hour is None else hour
    if hour < config.window-1 or hour > config.duration_h:
        raise HTTPException(422, f"Hour must be between {config.window-1} and {config.duration_h}")
    return config, engine, frame[frame.hour <= hour], hour


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "synthetic", "phase": 1}


@app.get("/api/reference")
def reference():
    return {"parameters": PARAMETER_INFO, "scenarios": SCENARIOS,
            "defaults": SessionConfig().model_dump(),
            "provenance": json.loads((ROOT / "docs" / "reference.json").read_text(encoding="utf-8"))}


@app.get("/api/sessions")
def sessions():
    with db() as connection:
        rows = connection.execute("SELECT id,created,config FROM sessions ORDER BY created DESC LIMIT 30").fetchall()
    return [{"id": r[0], "created": r[1], "config": json.loads(r[2])} for r in rows]


@app.post("/api/sessions", status_code=201)
def create_session(config: SessionConfig):
    with lock:
        engine = AnalyticalEngine(config)
        measurements = generate(config)
        frame = engine.score(measurements)
        session_id = uuid4().hex[:12]
        with db() as connection:
            connection.execute("INSERT INTO sessions VALUES (?,?,?,?)", (session_id, datetime.now(timezone.utc).isoformat(),
                               config.model_dump_json(), measurements.to_json(orient="records", double_precision=15)))
        if len(cache) >= 3:
            cache.pop(next(iter(cache)))
        cache[session_id] = (config, engine, frame)
    return {"id": session_id, "config": config.model_dump(), "rows": len(frame), "synthetic": True}


@app.get("/api/sessions/{session_id}/summary")
def summary(session_id: str, hour: int | None = None):
    config, engine, frame, hour = observed(session_id, hour)
    latest = frame.groupby("component_id", sort=False).tail(1)
    columns = ["component_id", "hour", "risk_probability", "action", "anomaly_flag", "drift_flag", "static_fail", "isolation_score", "correlation_drift", "covariance_drift", *PARAMETERS]
    counts = latest.action.value_counts().to_dict()
    alerts = frame[(frame.anomaly_flag | frame.drift_flag | frame.static_fail) & frame.ready]
    alerts = alerts.sort_values(["hour", "component_id"], ascending=[False, True]).head(60)
    lot = {"correlation": frame[[f"z_{p}" for p in PARAMETERS]].corr().to_numpy().tolist(),
           "mean_risk": float(latest.risk_probability.mean()), "drifted_components": int(latest.drift_flag.sum())}
    return {"id": session_id, "hour": hour, "config": config.model_dump(), "counts": counts,
            "components": records(latest[columns]), "alerts": records(alerts[columns]), "lot": lot,
            "baseline": engine.baseline_summary, "metrics": performance(frame, config),
            "model_version": engine.version, "split_ids": {**engine.split_ids, "demo": frame.component_id.unique().tolist()}}


@app.get("/api/sessions/{session_id}/components/{component_id}")
def component(session_id: str, component_id: str, hour: int | None = None):
    config, engine, frame, hour = observed(session_id, hour)
    group = frame[frame.component_id == component_id]
    if group.empty:
        raise HTTPException(404, "Component not found in this session")
    row = group.iloc[-1]
    forecast = engine.forecast(group)
    decision = costs(row.risk_probability, row.static_fail, config.false_negative_cost,
                     config.false_positive_cost, row.anomaly_flag or row.drift_flag, row.ready)
    reasons = []
    if row.static_fail:
        reasons.append("Measured parameter outside a configured demonstration specification limit.")
    if row.anomaly_flag:
        reasons.append(f"Isolation Forest score {row.isolation_score:.3f} exceeds healthy calibration threshold {engine.if_threshold:.3f}.")
    if row.drift_flag:
        reasons.append("Trailing residual correlation or covariance distance exceeds its healthy calibration threshold.")
    reasons.append(f"Calibrated synthetic abnormality probability {row.risk_probability:.1%}; cost-policy threshold {decision['threshold']:.1%}.")
    clean_columns = [c for c in group.columns if c not in ("scenario", "truth_active", "onset_h")]
    window = group.tail(config.window)[[f"z_{p}" for p in PARAMETERS]]
    return {"component_id": component_id, "hour": hour, "history": records(group[clean_columns]),
            "forecast": forecast, "decision": decision, "reasons": reasons, "explanation": engine.explain(row),
            "correlation": window.corr().to_numpy().tolist(), "covariance": window.cov().to_numpy().tolist(),
            "baseline_correlation": engine.baseline_corr.tolist(), "baseline_covariance": engine.baseline_cov.tolist(),
            "evaluation_truth": {"scenario": row.scenario, "active": bool(row.truth_active), "onset_h": int(row.onset_h)},
            "parameters": PARAMETER_INFO}


@app.get("/api/sessions/{session_id}/report")
def report(session_id: str, hour: int | None = None):
    result = summary(session_id, hour)
    result["notice"] = "All measurements are synthetic. Limits, failure prevalence and costs are demonstration assumptions. Not a qualification report."
    result["component_details"] = [component(session_id, c["component_id"], result["hour"]) for c in result["components"]]
    return Response(json.dumps(result, allow_nan=False), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="nirikshan-{session_id}-h{result["hour"]}.json"'})


@app.get("/api/sessions/{session_id}/measurements.csv")
def export_measurements(session_id: str, hour: int | None = None):
    _, _, frame, hour = observed(session_id, hour)
    return Response(frame.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="synthetic-{session_id}-h{hour}.csv"'})


@app.get("/api/sessions/{session_id}/performance.png")
def performance_plot(session_id: str, hour: int | None = None):
    config, _, frame, _ = observed(session_id, hour)
    metric = performance(frame, config)
    with plot_lock:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        matrix = np.array([[metric["tn"], metric["fp"]], [metric["fn"], metric["tp"]]])
        axes[0].imshow(matrix, cmap="Blues")
        for (i,j), value in np.ndenumerate(matrix):
            axes[0].text(j, i, str(value), ha="center", va="center", color="black")
        axes[0].set(xticks=[0,1], yticks=[0,1], xticklabels=["Continue/watch", "Hold/reject"],
                    yticklabels=["Normal", "Injected anomaly"], title="Observed synthetic component-hours")
        sensitivity = metric["cost_sensitivity"]
        axes[1].plot([s["cost_ratio"] for s in sensitivity], [s["fn"] for s in sensitivity], "o-", label="False negatives")
        axes[1].plot([s["cost_ratio"] for s in sensitivity], [s["fp"] for s in sensitivity], "o-", label="False positives")
        axes[1].set(xlabel="False-negative / false-positive cost", ylabel="Component-hours", title="Cost sensitivity")
        axes[1].legend()
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=150)
        plt.close(fig)
    return Response(stream.getvalue(), media_type="image/png")


DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(DIST / "index.html")
