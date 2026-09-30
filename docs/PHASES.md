# Nirikshan Netra: three-phase delivery plan

Baseline: all six pages of `TEAM_HUSTLERS-SIH-26_PS-170.pdf`, including the visual architecture on page 3, and both worksheets in `Base Data.xlsx`. Phase 1 is the scope of this implementation. Everything runs on synthetic data in every phase.

## Phase 1: analytical prototype — implemented

Deliver a working local demonstration from synthetic telemetry to inspectable decisions.

- Seeded NumPy/pandas generator for IDDQ, leakage and propagation delay; temperature/voltage stress; six scenarios.
- Schema validation, healthy-only normalization, trailing windows, deltas, rolling statistics and residual drift rates.
- Learned stress response using scikit-learn regression; stress-adjusted residuals.
- Isolation Forest dynamic anomaly detection with held-out healthy threshold calibration.
- Correlation and covariance drift against pooled healthy windows, plus component and lot outputs.
- statsmodels OLS trajectories with conditional 95% observation intervals and predicted limit crossings.
- XGBoost risk model, separate logistic probability calibration, native TreeSHAP explanations.
- Static-limit overrides and configurable C_FN/C_FP expected-cost decisions and sensitivity curves.
- React / Plotly dashboard, configuration, history replay, feature tables, forecast plots, matrices, model traces and observed evaluation results.
- FastAPI endpoints, SQLite session persistence, CSV and full JSON risk reports, matplotlib evaluation PNG.
- Tests for future-data isolation, independent components, expected algorithm behavior, policy boundaries, validation, persistence and exports.

Acceptance: a run can be created and reopened; each algorithm produces a numerical output; all three parameters and six scenarios are inspectable; forecasts do not read future observations; exports match the selected observation hour; implementation passes automated and browser checks.

## Phase 2: complete screening workflow — planned

- Engineer accounts and actual login, roles and review permissions; no cosmetic authentication screen.
- Parameter registry with editable units/specification limits and family-specific profiles.
- Configurable monitoring schedules; preview, start, pause, resume and complete a screening session.
- Simulated bench/chamber connector with incremental sample arrival, validation and disconnect recovery.
- Live backend streaming or polling, persistent alert lifecycle, acknowledgment and review notes.
- Component and lot report review, scheduled risk reports and final screening disposition with reviewer sign-off.
- Immutable stored model artifacts and versioned per-run outputs, rather than deterministic retraining on cache misses.

Acceptance: the complete slide-2 user flow works from login through simulated chamber execution, live alerts, report analysis and reviewer decisions. Sensors remain simulated.

## Phase 3: validation and deployment — planned

- Multi-seed and held-out-lot experiments; new anomaly shapes, missing samples and noise regimes; calibration reliability and threshold stability.
- Forecast backtests, prediction-interval coverage, early-warning lead time and false-alert rates per component/session.
- Cost-ratio and prevalence sensitivity; validate correlation/covariance distances and minimum-window requirements.
- PostgreSQL persistence, migration from SQLite, Docker packaging and reproducible deployment using the slide's stack.
- Git/GitHub CI, model artifact fingerprints, run audit trails, API access controls and performance/load tests.
- Rehearsable demonstration script, deployment guide and evidence tied to the declared synthetic assumptions.

Acceptance: reproducible deployment, measured performance across held-out synthetic conditions, durable auditing and a documented demonstration. Any claims about real components require real validation outside this synthetic prototype.

## Presentation coverage

| Source | Requirement | Phase 1 output | Remaining phase |
|---|---|---|---|
| Slide 1 | SIH26170, component burn-in anomaly detection | Named local application and project | — |
| Slide 2 | Stress-conditioned detection | Learned expected response, residual / z-score plots | Wider profiles in 2–3 |
| Slide 2 | Dynamic anomalies | Isolation score and calibrated threshold | Broader validation in 3 |
| Slide 2 | Trajectory and future-risk range | OLS mean / observation interval / limit crossing | Coverage backtests in 3 |
| Slide 2 | Multivariate relationships | Correlation and covariance matrices/distances | Window robustness in 3 |
| Slide 2 | Login / schedule / bench / chamber | Configured run and synthetic replay only | Full operational flow in 2 |
| Slide 2 | Alerts and report analysis | Calculated flags, inspector and JSON/CSV reports | Persistent alert review in 2 |
| Slide 3 | Ingestion / normalization / windowing | Schema guard, training-only scaler, causal features | Streaming ingestion in 2 |
| Slide 3 | Isolation Forest, regression, correlation drift | All execute in the backend | Experiments in 3 |
| Slide 3 | XGBoost + SHAP-style justification | Actual XGBoost and native TreeSHAP contributions | Validation in 3 |
| Slide 3 | Component and lot risk nodes | Component table and lot aggregation | Final review flow in 2 |
| Slide 3 | Static limit + risk + C_FN/C_FP | Expected-cost policy with static override | Sensitivity validation in 3 |
| Slide 3 | React, FastAPI, Plotly, matplotlib | Working UI/API and figure export | — |
| Slide 3 | PostgreSQL / SQLite + Docker | SQLite, as explicitly allowed by architecture | PostgreSQL and Docker in 3 |
| Slide 4 | No proprietary data; transparent split | Synthetic-only generation; independent component groups | Broader experiments in 3 |
| Slide 5 | Earlier warning and traceable decisions | Pre-limit HOLD, features and reconstructed score | Lead-time / audit study in 3 |
| Slide 6 | Research context | Reference notes and Excel cell provenance | Extend only as required |

Extra tooling is limited to Vite (React build/dev server), Uvicorn (FastAPI ASGI server), pytest/httpx (verification), and transitive dependencies. Python's standard-library sqlite3 avoids an additional ORM. No external telemetry, cloud service, LLM, or new ML framework is required.
