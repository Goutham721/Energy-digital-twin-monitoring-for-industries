"""Unit tests for src.data_generator."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data_generator import generate_dataset


@pytest.fixture(scope="module")
def small_dataset():
    return generate_dataset(num_records=1000, save=False)


def test_dataset_not_empty(small_dataset):
    assert len(small_dataset) > 0


def test_dataset_has_required_columns(small_dataset):
    required = {
        "timestamp", "machine_id", "machine_type", "machine_status",
        "load_percentage", "production_rate", "temperature", "pressure",
        "vibration", "efficiency", "operating_hours", "maintenance_status",
        "ambient_temperature", "energy_consumption_kwh", "power_kw",
        "energy_cost", "carbon_emission", "anomaly_flag", "fault_type",
    }
    assert required.issubset(set(small_dataset.columns))


def test_energy_consumption_non_negative(small_dataset):
    assert (small_dataset["energy_consumption_kwh"] >= 0).all()


def test_efficiency_within_bounds(small_dataset):
    assert (small_dataset["efficiency"] >= 0.3).all()
    assert (small_dataset["efficiency"] <= 1.0).all()


def test_machine_ids_are_known(small_dataset):
    assert set(small_dataset["machine_id"].unique()).issubset(set(config.MACHINE_IDS))


def test_reproducibility_with_seed():
    df1 = generate_dataset(num_records=500, seed=123, save=False)
    df2 = generate_dataset(num_records=500, seed=123, save=False)
    assert df1["energy_consumption_kwh"].tolist() == df2["energy_consumption_kwh"].tolist()
