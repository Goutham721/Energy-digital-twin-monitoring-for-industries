"""Unit tests for src.energy_optimizer, energy_cost, carbon_calculator."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generator import generate_dataset
from src.energy_optimizer import optimize_plant_load
from src.energy_cost import calculate_cost, cost_summary
from src.carbon_calculator import calculate_emissions, emissions_summary


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(num_records=2000, save=False)


@pytest.fixture
def demo_machine_states():
    return [
        {"machine_id": "Motor_1", "machine_type": "Motor", "rated_kw": 55,
         "efficiency": 0.9, "load_percentage": 80, "status": "Running"},
        {"machine_id": "Compressor_1", "machine_type": "Compressor", "rated_kw": 75,
         "efficiency": 0.85, "load_percentage": 70, "status": "Running"},
        {"machine_id": "HVAC_1", "machine_type": "HVAC", "rated_kw": 60,
         "efficiency": 0.8, "load_percentage": 65, "status": "Running"},
    ]


def test_optimizer_reduces_or_maintains_energy(demo_machine_states):
    result = optimize_plant_load(demo_machine_states)
    assert result["optimized_energy_kwh"] <= result["current_energy_kwh"] + 1e-6


def test_optimizer_savings_non_negative(demo_machine_states):
    result = optimize_plant_load(demo_machine_states)
    assert result["savings_percent"] >= 0


def test_optimizer_no_running_machines_handled_gracefully():
    result = optimize_plant_load([{"machine_id": "X", "status": "Stopped", "rated_kw": 10}])
    assert result["current_energy_kwh"] == 0.0
    assert result["recommendations"] == []


def test_cost_calculation_flat(dataset):
    result = calculate_cost(dataset, tariff_mode="flat")
    assert "energy_cost" in result.columns
    assert (result["energy_cost"] >= 0).all()


def test_cost_summary_keys(dataset):
    result = calculate_cost(dataset, tariff_mode="time_of_use")
    summary = cost_summary(result)
    assert set(["total_cost", "hourly_avg_cost", "daily_avg_cost", "monthly_estimated_cost"]).issubset(summary)


def test_carbon_emission_non_negative(dataset):
    result = calculate_emissions(dataset)
    assert (result["carbon_emission"] >= 0).all()


def test_emissions_summary_keys(dataset):
    result = calculate_emissions(dataset)
    summary = emissions_summary(result)
    assert "total_emission_kg" in summary
