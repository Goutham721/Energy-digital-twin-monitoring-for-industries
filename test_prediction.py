"""Unit tests for src.train_energy_model, anomaly_detection, predictive_maintenance."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generator import generate_dataset
from src.train_energy_model import predict_energy, train_and_select_best
from src.anomaly_detection import detect_anomalies, train_anomaly_model
from src.predictive_maintenance import predict_health, train_maintenance_model


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(num_records=3000, save=False)


@pytest.fixture(scope="module")
def trained_model(dataset):
    return train_and_select_best(df=dataset, save=False)


def test_model_training_returns_metrics(trained_model):
    assert "test_metrics" in trained_model
    assert trained_model["test_metrics"]["R2"] is not None


def test_model_r2_reasonable(trained_model):
    # a physically-motivated dataset should be highly predictable
    assert trained_model["test_metrics"]["R2"] > 0.5


def test_predict_energy_non_negative(dataset):
    train_and_select_best(df=dataset, save=True)
    row = dataset.iloc[0]
    input_dict = {
        "load_percentage": row["load_percentage"], "production_rate": row["production_rate"],
        "temperature": row["temperature"], "pressure": row["pressure"],
        "vibration": row["vibration"], "efficiency": row["efficiency"],
        "operating_hours": row["operating_hours"], "ambient_temperature": row["ambient_temperature"],
        "hour": row["hour"], "day_of_week": row["day_of_week"], "is_weekend": row["is_weekend"],
        "machine_type": row["machine_type"], "machine_status": row["machine_status"],
        "maintenance_status": row["maintenance_status"],
    }
    pred = predict_energy(input_dict)
    assert pred >= 0


def test_anomaly_detection_flags_column(dataset):
    train_anomaly_model(df=dataset, save=True)
    result = detect_anomalies(dataset)
    assert "anomaly_flag_detected" in result.columns
    assert set(result["anomaly_flag_detected"].dropna().unique()).issubset({0, 1})


def test_predictive_maintenance_conditions(dataset):
    train_maintenance_model(df=dataset, save=True)
    result = predict_health(dataset)
    assert "predicted_condition" in result.columns
    assert result["risk_score"].between(0, 100).all()
