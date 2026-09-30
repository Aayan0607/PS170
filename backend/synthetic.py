"""Entirely synthetic measurements. Reference values are context, not training rows."""
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

PARAMETERS = ["iddq_ma", "leakage_ua", "delay_ns"]
PARAMETER_INFO = [
    {"key": "iddq_ma", "name": "Quiescent supply current", "unit": "mA", "lower": 5.0, "upper": 22.0},
    {"key": "leakage_ua", "name": "Leakage current", "unit": "µA", "lower": 0.05, "upper": 3.0},
    {"key": "delay_ns", "name": "Propagation delay", "unit": "ns", "lower": 25.0, "upper": 100.0},
]
SCENARIOS = ["healthy", "stress_only", "gradual_degradation", "relationship_drift", "intermittent_spikes", "sudden_shift"]


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    seed: int = Field(default=170, ge=0, le=1_000_000)
    components: int = Field(default=24, ge=6, le=60)
    duration_h: int = Field(default=168, ge=72, le=336)
    temperature_c: float = Field(default=125, ge=100, le=140)
    voltage_v: float = Field(default=4.0, ge=3.6, le=4.4)
    cycle_h: int = Field(default=24, ge=8, le=48)
    window: int = Field(default=24, ge=12, le=48)
    horizon_h: int = Field(default=24, ge=6, le=48)
    false_negative_cost: float = Field(default=20, ge=1, le=1000)
    false_positive_cost: float = Field(default=1, ge=0.1, le=100)

    @model_validator(mode="after")
    def enough_history(self):
        if self.window >= self.duration_h:
            raise ValueError("Window must be shorter than the screening duration")
        return self


def stress_profile(hours, config, extra=0.0):
    h = np.asarray(hours, dtype=float)
    temperature = config.temperature_c + (6 + extra) * np.sin(2 * np.pi * h / config.cycle_h)
    voltage = config.voltage_v + 0.06 * np.cos(2 * np.pi * h / config.cycle_h)
    return temperature, voltage


def generate(config: SessionConfig, count=None, role="demo", healthy_only=False):
    """Independent RNG streams and component IDs for each train/calibration/test role."""
    roles = {"baseline": 1, "threshold": 2, "risk_train": 3, "risk_calibration": 4, "demo": 5}
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, roles[role]]))
    count = count or config.components
    frames = []
    hours = np.arange(config.duration_h + 1, dtype=float)
    for i in range(count):
        scenario = SCENARIOS[i % len(SCENARIOS)] if not healthy_only else SCENARIOS[i % 2]
        temperature, voltage = stress_profile(hours, config, extra=5 if scenario == "stress_only" else 0)
        dt, dv = temperature - 125, voltage - 4
        expected = np.column_stack([12 + .055 * dt + 2 * dv,
                                    .65 + .012 * dt + .0003 * dt**2 + .2 * dv,
                                    60 - .11 * dt - 6 * dv])
        common = rng.normal(size=len(hours))
        noise = .8 * common[:, None] + .6 * rng.normal(size=(len(hours), 3))
        scales = np.array([.55, .045, 1.1])
        values = expected + noise * scales + rng.normal(size=3) * scales * .12
        onset = int(rng.integers(32, 57))
        active = np.zeros(len(hours), dtype=bool)
        age = np.maximum(hours - onset, 0)
        if scenario == "gradual_degradation":
            active = hours >= onset
            values += age[:, None] * np.array([.082, .022, .30]) * rng.uniform(.85, 1.15)
        elif scenario == "relationship_drift":
            active = hours >= onset
            # Reverse a relationship without shifting each parameter's marginal mean.
            values[active, 2] = expected[active, 2] + (-.8 * common[active] + .6 * rng.normal(size=active.sum())) * scales[2]
        elif scenario == "intermittent_spikes":
            active = (hours >= onset) & ((hours.astype(int) - onset) % 19 < 3)
            values[active] += np.array([5.5, 1.1, 16])
        elif scenario == "sudden_shift":
            active = hours >= onset
            values[active] += np.array([4.2, .85, 12])
        frame = pd.DataFrame(values, columns=PARAMETERS)
        frame["hour"] = hours
        frame["temperature_c"] = temperature
        frame["voltage_v"] = voltage
        frame["stress_extra_c"] = 5 if scenario == "stress_only" else 0
        frame["component_id"] = f"{role.upper()}-{i + 1:03d}"
        frame["scenario"] = scenario
        frame["truth_active"] = active
        frame["onset_h"] = onset if scenario not in SCENARIOS[:2] else -1
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_measurements(frame):
    required = ["component_id", "hour", "temperature_c", "voltage_v", *PARAMETERS]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    numeric = frame[["hour", "temperature_c", "voltage_v", *PARAMETERS]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Measurements must be finite; missing values are not silently imputed")
    if frame.duplicated(["component_id", "hour"]).any():
        raise ValueError("Duplicate component and timestamp")
    for _, group in frame.groupby("component_id", sort=False):
        if not np.all(np.diff(group.hour) == 1):
            raise ValueError("Expected ordered hourly samples per component")
    return frame
