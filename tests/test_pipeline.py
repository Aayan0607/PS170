import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from backend.synthetic import SessionConfig, generate, validate_measurements, PARAMETERS
from backend.engine import AnalyticalEngine, costs, performance
from backend import main


@pytest.fixture(scope="module")
def pipeline():
    config = SessionConfig()
    engine = AnalyticalEngine(config)
    raw = generate(config)
    return config, engine, raw, engine.score(raw)


def test_reproducible_synthetic_data_and_disjoint_streams():
    c = SessionConfig(components=6)
    pd.testing.assert_frame_equal(generate(c), generate(c))
    assert not generate(c).iddq_ma.equals(generate(c,role="risk_train").iddq_ma)
    assert generate(c).scenario.nunique() == 6


def test_schema_rejects_missing_nonfinite_duplicate_and_disordered_rows():
    frame = generate(SessionConfig(components=6))
    for bad in [frame.drop(columns="hour"), pd.concat([frame, frame.iloc[:1]]), frame.iloc[::-1]]:
        with pytest.raises(ValueError):
            validate_measurements(bad)
    frame.loc[0, "iddq_ma"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_measurements(frame)


@pytest.mark.parametrize("bad", [{"seed":-1},{"components":0},{"window":1},{"false_negative_cost":float('nan')},{"voltage_v":0},{"unknown":True}])
def test_config_validation(bad):
    with pytest.raises(ValidationError):
        SessionConfig(**bad)


def test_no_shared_component_ids(pipeline):
    _, e, raw, _ = pipeline
    groups = [set(v) for v in e.split_ids.values()] + [set(raw.component_id)]
    assert all(not a & b for i,a in enumerate(groups) for b in groups[i+1:])
    assert not set(e.risk_columns) & {"truth_active", "scenario", "onset_h", "component_id", "hour"}


def test_future_and_truth_cannot_change_past_outputs(pipeline):
    _, e, raw, full = pipeline
    prefix = raw[raw.hour <= 80].copy()
    expected = full[full.hour <= 80].reset_index(drop=True)
    scored = e.score(prefix)
    pd.testing.assert_frame_equal(scored, expected)
    changed = raw.copy()
    changed.loc[changed.hour>80, PARAMETERS] *= 10
    changed["scenario"] = "fake"
    changed["truth_active"] = False
    changed["onset_h"] = 0
    altered = e.score(changed)
    columns = e.risk_columns + ["risk_probability", "action", "drift_flag", "anomaly_flag"]
    pd.testing.assert_frame_equal(altered[altered.hour<=80][columns].reset_index(drop=True), expected[columns])
    a = e.forecast(scored[scored.component_id=="DEMO-003"])
    b = e.forecast(altered[(altered.component_id=="DEMO-003") & (altered.hour<=80)])
    assert a == b


def test_warmup_never_has_probability_or_drift(pipeline):
    c, _, _, f = pipeline
    warm = f[f.hour<c.window-1]
    assert warm.risk_probability.isna().all()
    assert not warm.drift_flag.any()
    assert set(warm.action) == {"WARMUP"}


def test_tree_shap_reconstructs_margin_and_calibration(pipeline):
    _, e, _, f = pipeline
    row = f[(f.component_id=="DEMO-003") & (f.hour==96)].iloc[0]
    explanation = e.explain(row)
    margin = e.risk.predict(row[e.risk_columns].astype(float).to_frame().T, output_margin=True)[0]
    assert explanation["bias"] + sum(x["contribution"] for x in explanation["contributions"]) == pytest.approx(float(margin), abs=1e-5)
    assert explanation["calibrated_probability"] == pytest.approx(row.risk_probability,abs=1e-6)


def test_forecast_horizon_and_intervals(pipeline):
    c, e, _, f = pipeline
    prediction = e.forecast(f[(f.component_id=="DEMO-003") & (f.hour<=120)])
    assert len(prediction["parameters"]) == 3
    for p in prediction["parameters"]:
        assert p["hours"][0] == 121 and p["hours"][-1] == 120+c.horizon_h
        assert np.isfinite(np.array([p["mean"],p["lower"],p["upper"]])).all()
        assert np.all(np.array(p["lower"]) < p["mean"])
        assert np.all(np.array(p["upper"]) > p["mean"])
    assert prediction["parameters"][1]["possible_crossing_h"] is not None


def test_cost_policy_and_static_override():
    assert costs(.1,False,1,1)["action"] == "CONTINUE"
    assert costs(.1,False,20,1)["action"] == "HOLD"
    assert costs(.01,True,20,1)["action"] == "REJECT"
    assert costs(.01,False,20,1,warning=True)["action"] == "WATCH"
    assert costs(.5,False,1,1)["action"] == "HOLD"


def test_injected_drift_and_early_warning_are_detected(pipeline):
    _, _, _, f = pipeline
    drift = f[(f.scenario=="relationship_drift") & (f.hour>=96)]
    assert drift.drift_flag.mean() > .8
    healthy = f[(f.scenario=="stress_only") & f.ready]
    assert healthy.anomaly_flag.mean() < .05
    progressive = f[f.component_id=="DEMO-003"]
    first_hold = progressive[progressive.action=="HOLD"].hour.min()
    first_breach = progressive[progressive.static_fail].hour.min()
    assert first_hold < first_breach < 168


def test_metrics_are_calculated_from_observed_truth(pipeline):
    c, _, _, f = pipeline
    m = performance(f[f.hour<=96], c)
    assert sum(m[k] for k in ["tn","fp","fn","tp"]) == m["samples"] == 24*(96-c.window+2)
    assert m["realized_cost"] == m["fp"]*c.false_positive_cost + m["fn"]*c.false_negative_cost
    assert all(a["holds_or_rejects"]<=b["holds_or_rejects"] for a,b in zip(m["cost_sensitivity"],m["cost_sensitivity"][1:]))


def test_api_persistence_exports_and_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", tmp_path/"test.sqlite3")
    monkeypatch.setattr(main, "cache", {})
    client = TestClient(main.app)
    assert client.get('/api/health').json()["mode"] == "synthetic"
    assert client.get('/api/reference').status_code == 200
    assert client.post('/api/sessions',json={"components":2}).status_code == 422
    run = client.post('/api/sessions',json={"components":6})
    assert run.status_code == 201, run.text
    sid = run.json()["id"]
    path = f'/api/sessions/{sid}'
    s = client.get(path+'/summary?hour=96')
    assert s.status_code == 200
    data = s.json()
    assert len(data["components"]) == 6
    assert all(r["hour"]==96 for r in data["components"])
    assert client.get(path+'/summary?hour=500').status_code == 422
    assert client.get(path+'/summary?hour=3').status_code == 422
    assert client.get('/api/sessions/missing/summary').status_code == 404
    assert client.get(path+'/components/missing?hour=96').status_code == 404
    detail = client.get(path+'/components/DEMO-003?hour=96').json()
    assert all(r["hour"]<=96 for r in detail["history"])
    assert "truth_active" not in detail["history"][0]
    csv = client.get(path+'/measurements.csv?hour=96')
    assert csv.status_code == 200 and len(csv.text.splitlines())==6*97+1
    report = client.get(path+'/report?hour=96')
    assert report.status_code == 200, report.text
    assert len(report.json()["component_details"]) == 6
    assert client.get(path+'/performance.png?hour=96').content.startswith(b'\x89PNG')
    main.cache.clear()
    reloaded = client.get(path+'/summary?hour=96').json()
    assert [r['action'] for r in data['components']] == [r['action'] for r in reloaded['components']]
    np.testing.assert_allclose([r['risk_probability'] for r in data['components']], [r['risk_probability'] for r in reloaded['components']],rtol=1e-5)
    assert client.get('/api/sessions').json()[0]['id'] == sid
