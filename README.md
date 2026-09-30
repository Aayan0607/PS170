# Nirikshan Netra

Phase 1 prototype for **SIH26170 — AI-Driven Anomaly Detection in Component Burn-In & Screening**. Implemented from the six-slide TEAM_HUSTLERS PDF. All telemetry and model-training data are synthetic.

## Run locally

The project is in the PS170 folder. With dependencies and the frontend already built:

```powershell
.\start.ps1
```

Open http://127.0.0.1:8000. Interactive API documentation: http://127.0.0.1:8000/docs.

If PowerShell script execution is restricted, run the underlying command directly:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### First-time setup on another machine

Python 3.12 and a Node version supported by Vite 7 (20.19+ or 22.12+) are required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm.cmd ci
npm.cmd run build
cd ..
.\start.ps1
```

The frontend is served by FastAPI. No CDN, cloud API, credentials or hardware are needed after installation. For development, run the API on port 8000 and `npm.cmd run dev` inside frontend in a second terminal; Vite proxies `/api`.

## Demonstrate the prototype

1. Click **Launch demonstration run** or **New run**. Default: seed 170, 24 components, 168 h, 125 °C and 4 V. Model fitting takes a few seconds on a standard CPU.
2. Select **DEMO-003** (gradual degradation). At hour 96 the dashboard shows stress residuals, forecasts, anomaly score, drift and risk before its eventual specification breach.
3. In **Signals & forecast**, switch between IDDQ, leakage and delay. Move the observation hour to 120 to inspect possible future limit crossings.
4. Select **DEMO-004** and open **Detection & drift**. Inspect changed relationships despite parameters remaining within limits. Compare **DEMO-002** for normal stress variation.
5. Open **Decision trace** to inspect actual TreeSHAP contributions, calibration and the expected-cost calculation. **Validation** shows mistakes as well as successful detections.
6. Replay the history, download CSV / JSON risk report, and create another run with a different seed or cost ratio. Saved runs remain in SQLite.

## Three phases

| Phase | Scope | Status |
|---|---|---|
| 1 | Synthetic analytical pipeline, all slide algorithms, inspectable dashboard/API and exports | Implemented |
| 2 | Login, parameter registry, schedules, simulated bench lifecycle, live alerts and report review | Planned |
| 3 | Broader validation, forecast backtests, PostgreSQL, Docker and deployment/audit hardening | Planned |

See [complete phase plan](docs/PHASES.md), [methods and limitations](docs/METHODS.md), and [Excel reference provenance](docs/reference.json).

## Stack

React, FastAPI, Python, NumPy, pandas, scikit-learn, XGBoost, SciPy, statsmodels, Plotly, matplotlib and SQLite — all named in the presentation's architecture. Vite builds React, Uvicorn serves FastAPI, and pytest/httpx verify behavior. PostgreSQL and Docker are reserved for Phase 3. No extra ML framework is used.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm.cmd run build
```

Tests cover synthetic reproducibility, malformed measurements, independent splits, future-data isolation, warmup, SHAP additivity, forecasts, policy boundaries, drift detection, early warning, evaluation arithmetic, API behavior, persistence and exports.

## Layout

- `backend/synthetic.py`: schema, assumptions and generator.
- `backend/engine.py`: stress, anomaly, drift, trajectory, risk and decision engines.
- `backend/main.py`: API, SQLite persistence, report and plot endpoints.
- `frontend/src/`: React dashboard and styles.
- `tests/`: automated analytical and API verification.
- `docs/`: source mapping, phase plan, methodology and verification record.
- `data/`: generated local session database.
- `reference-extract/`: read-only source inspection outputs.

The probability describes **synthetic injected abnormality**, not a validated mission-failure probability. Sensor values, specification limits, prevalence and costs are demonstration assumptions. The prototype has no login or physical bench connection yet. OLS intervals are conditional estimates whose coverage still needs Phase 3 backtesting. Original PDF and workbook remain unchanged.
