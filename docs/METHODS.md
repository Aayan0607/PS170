# Analytical methods and boundaries

## Data and reference handling

`reference.json` records exact Excel ranges, observations, use and limitations. `Sheet3` is empty; `Sheet1` contains adjacent blocks from several reports and unrelated component families. These are reference tables, not a clean time-series dataset. The supplied files are unchanged.

Defaults of 125 °C, 168 h and 4 V follow Sheet1!AC96:AE96. The separate 300 h specification entry is not relabeled as 168 h. Leakage examples concern a 50 V capacitor; they are not applied as specifications for the synthetic 4 V device. IDDQ values, limits, correlations, noise, slopes and scenario prevalence are declared assumptions.

Some workbook values need source verification: transistor vendor A reports 29 failures / 55,220 h but lists 5.25 per million h; the arithmetic would give about 525 per million h, consistent with MTBF near 1,905 h. Vendor B has a similar discrepancy. Headings also say `1250 C`. The implementation does not use these entries as probabilities or silently repair the workbook.

The NASA record at https://ntrs.nasa.gov/citations/19840013934 was verified as *NASA flight electronics environmental stress screening survey* (December 1983). Two other NTRS links returned 403 when checked. No unverified web content is used to calibrate numeric inputs.

## Generator

Three channels: IDDQ in mA, leakage in microamps and propagation delay in ns. One sample per hour, including hour zero. Temperature and supply follow deterministic known cycles. Expected values respond to centered temperature, squared temperature and voltage; correlated Gaussian measurement noise and small component offsets are added. A stress-only cohort has a larger temperature amplitude and no degradation.

Six equally represented scenarios rotate by component: healthy, stress-only, gradual degradation, relationship drift, intermittent spikes and sudden shift. Injections begin at a seeded hour from 32 through 56. Relationship drift reverses delay's loading on shared noise, keeping approximately the same marginal mean and variance. Labels mark only active injected behavior, not future mission failure. Abrupt injections have no artificial precursor.

## Independent splits

NumPy SeedSequence combines the user seed with separate role IDs:

| Group | Components | Purpose |
|---|---:|---|
| Baseline | 36 healthy/stress-only | Stress regression, residual scaler, pooled healthy covariance/correlation, Isolation Forest |
| Threshold | 24 healthy/stress-only | 99th-percentile distance and Isolation Forest thresholds |
| Risk train | 72 mixed | XGBoost classifier |
| Risk calibration | 36 mixed | Logistic calibration of XGBoost margins |
| Demo | 6–60 mixed | Display and evaluation only |

Distinct component identities and random streams prevent within-component random train/test row splitting. All streams still share the same generator family: this is not evidence of robustness to an unseen physical process. No demo labels, component IDs, scenario IDs or injection times are feature inputs. Only evaluation reads demo labels.

## Algorithms

1. **Stress-conditioned regression:** scikit-learn LinearRegression predicts three channels from `[T−125, (T−125)^2, V−4, (T−125)(V−4)]` fitted on baseline healthy components. Measured minus expected gives a residual. StandardScaler fits only baseline residuals. Coefficients and scales are returned by the API.
2. **Feature engine:** pandas trailing windows, default 24 samples, produce raw deltas, rolling mean/std and residual linear slope. NumPy least-squares slope uses centered window time. No centered windows, backfill or future interpolation. The first W−1 samples have no rolling risk output.
3. **Isolation Forest:** 120 trees, 256 sampled observations per tree, trained on the three standardized healthy residuals. Display `−score_samples`; a larger value is more anomalous. Threshold is the 99th percentile on the separate healthy-threshold group. These scores are not probabilities.
4. **Relationship drift:** Pearson correlation matrix and sample covariance matrix of each trailing standardized-residual window. Correlation distance = Frobenius norm of `R_window−R_healthy`. Covariance distance = `||C_window−C_healthy||F / ||C_healthy||F`. Calibration uses identically constructed windows from independent healthy components, making distances and thresholds comparable. Both distances are reported, with a flag if either exceeds its threshold. The combined false-positive rate need not equal 1%.
5. **Trajectory regression:** statsmodels OLS fits a linear trend to each channel's last W stress residuals. Add the known future stress response to obtain raw-unit forecasts. `get_prediction().summary_frame(alpha=.05)` provides conditional 95% observation intervals. Report first mean crossing and first possible crossing where the interval reaches a limit. This assumes linear residual evolution, independent homoscedastic errors and a known stress schedule; stress-model uncertainty and scenario changes are not included. Interval coverage is not yet validated. No unsupported forecast accuracy claim is made.
6. **XGBoost risk fusion:** 100 depth-3 trees combine Isolation Forest score, maximum absolute current residual z-score, correlation distance, covariance distance, maximum absolute projected residual z and minimum normalized specification margin. The target is active injected abnormality. Separate LogisticRegression maps the raw XGBoost margin to a synthetic calibrated probability. Calibration prevalence reflects this generator and may not match a field population.
7. **Explainability:** XGBoost native `pred_contribs=True` computes TreeSHAP values. Bias + contributions reconstruct raw log-odds. Raw probability is SciPy's expit(margin). The displayed final probability is expit(calibration slope × margin + calibration intercept). Contributions explain the uncalibrated classifier and are not presented as additive final-probability contributions or physical causes.
8. **Cost policy:** continue cost = p × C_FN; hold cost = (1−p) × C_FP. HOLD when p ≥ C_FP/(C_FN+C_FP). Measured limit breaches override to REJECT. A statistical flag below the cost threshold becomes WATCH; otherwise CONTINUE. Early insufficient windows show WARMUP. CONTINUE is a monitoring recommendation, not final certification. Defaults are C_FN=20, C_FP=1; the validation view recomputes false positives/negatives across six ratios.

## Outputs and persistence

The observation-hour slider limits all API response histories, metric calculations and reports. Forecasts only see a prefix of a component. Future ground truth is never plotted as a forecast. Labels can be revealed separately for demonstration.

SQLite stores run configuration, creation time and synthetic raw measurements. On a process restart or cache eviction, the exact configuration is retrained deterministically with the installed dependency versions. Dependency locks are saved, but this is not yet immutable model-artifact auditing; that is a Phase 2 deliverable. Up to three fitted runs are retained in memory. The API is a local prototype without authentication and binds to localhost by default.

CSV exports include full feature and algorithm output columns through the selected hour, plus explicitly named synthetic evaluation labels. JSON reports contain configuration, model/split metadata, lot results, alerts, metrics, all component histories, predictions, matrices, contributions and cost reasons. Reports are JSON to keep every numeric algorithm output inspectable; formatted reviewer reports belong to Phase 2. matplotlib generates the downloadable confusion/sensitivity figure.

Metrics count observed held-out component-hours after warmup. Windows overlap. Precision/recall/F1 describe HOLD/REJECT decisions against active injection labels; Brier score evaluates synthetic abnormality probabilities. Cost sensitivity changes the classification threshold, not model fitting. False positives and false negatives remain visible.
