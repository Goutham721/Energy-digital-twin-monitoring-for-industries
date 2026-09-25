"""
config.py
Central configuration for the AI-Driven Smart Energy Digital Twin project.
All paths are relative to the project root so the app is portable
across Windows/Linux/Mac.
"""

from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# ROOT PATHS
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent

DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FILE NAMES
# ---------------------------------------------------------------------------
DATASET_PATH: Path = DATA_DIR / "industrial_energy_data.csv"
MACHINE_DATA_PATH: Path = DATA_DIR / "machine_data.csv"

BEST_MODEL_PATH: Path = MODELS_DIR / "best_energy_model.pkl"
PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.pkl"
MODEL_METRICS_PATH: Path = MODELS_DIR / "model_metrics.json"
ANOMALY_MODEL_PATH: Path = MODELS_DIR / "anomaly_model.pkl"
MAINTENANCE_MODEL_PATH: Path = MODELS_DIR / "maintenance_model.pkl"
FORECAST_MODEL_PATH: Path = MODELS_DIR / "forecast_model.pkl"

DATABASE_PATH: Path = DATA_DIR / "digital_twin.db"

# ---------------------------------------------------------------------------
# SIMULATION PARAMETERS
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
DEFAULT_NUM_RECORDS: int = 60_000          # 50,000 - 100,000 required
SIMULATION_START_DATE: str = "2024-01-01"
SIMULATION_FREQ_MINUTES: int = 15          # data granularity

MACHINES: dict[str, dict] = {
    "Motor_1":        {"type": "Motor",       "rated_kw": 55,  "base_eff": 0.92},
    "Motor_2":        {"type": "Motor",       "rated_kw": 45,  "base_eff": 0.90},
    "Conveyor_1":      {"type": "Conveyor",    "rated_kw": 18,  "base_eff": 0.88},
    "Compressor_1":    {"type": "Compressor",  "rated_kw": 75,  "base_eff": 0.85},
    "Pump_1":          {"type": "Pump",        "rated_kw": 30,  "base_eff": 0.87},
    "HVAC_1":          {"type": "HVAC",        "rated_kw": 60,  "base_eff": 0.80},
    "Boiler_1":        {"type": "Boiler",      "rated_kw": 90,  "base_eff": 0.83},
    "Cooling_System_1": {"type": "Cooling",    "rated_kw": 40,  "base_eff": 0.86},
    "Production_Line_1": {"type": "ProductionLine", "rated_kw": 100, "base_eff": 0.89},
    "Lighting_1":      {"type": "Lighting",    "rated_kw": 12,  "base_eff": 0.95},
}

MACHINE_IDS = list(MACHINES.keys())

# ---------------------------------------------------------------------------
# ENERGY COST / TARIFF DEFAULTS
# ---------------------------------------------------------------------------
DEFAULT_FLAT_TARIFF_PER_KWH: float = 0.14          # currency units / kWh
PEAK_TARIFF_PER_KWH: float = 0.24
OFF_PEAK_TARIFF_PER_KWH: float = 0.09
PEAK_HOURS: tuple[int, int] = (9, 21)              # 9am - 9pm considered peak

# ---------------------------------------------------------------------------
# CARBON EMISSION DEFAULTS
# ---------------------------------------------------------------------------
DEFAULT_EMISSION_FACTOR_KG_PER_KWH: float = 0.45   # configurable assumption

# ---------------------------------------------------------------------------
# ML / MODELING
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "energy_consumption_kwh"

NUMERIC_FEATURES: list[str] = [
    "load_percentage",
    "production_rate",
    "temperature",
    "pressure",
    "vibration",
    "efficiency",
    "operating_hours",
    "ambient_temperature",
    "hour",
    "day_of_week",
    "is_weekend",
]

CATEGORICAL_FEATURES: list[str] = [
    "machine_type",
    "machine_status",
    "maintenance_status",
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TEST_SIZE: float = 0.15
VALIDATION_SIZE: float = 0.15

# ---------------------------------------------------------------------------
# ANOMALY DETECTION
# ---------------------------------------------------------------------------
ANOMALY_CONTAMINATION: float = 0.03

# ---------------------------------------------------------------------------
# APP METADATA
# ---------------------------------------------------------------------------
APP_TITLE: str = "AI-Driven Smart Energy Digital Twin for Industrial Plants"
APP_SUBTITLE: str = "A software-based Digital Twin and AI simulation framework for industrial energy management."
DISCLAIMER: str = (
    "This is an academic, fully software-based simulation. All plant, machine, "
    "and sensor data shown in this application is SYNTHETIC / SIMULATED. "
    "No real hardware, IoT devices, or factory deployments are involved."
)
