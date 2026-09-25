# AI-Driven Smart Energy Digital Twin for Industrial Plants

A **software-only** Digital Twin and AI simulation framework for industrial
energy management, built as a final-year academic project.

> **Important:** This is a fully software-based simulation. There is **no**
> Arduino, ESP32, Raspberry Pi, PLC, physical sensor, or IoT hardware
> anywhere in this project. All plant, machine, and energy data is
> **synthetic**, generated using physically-motivated mathematical
> relationships (see `src/data_generator.py`).

---

## 1. What this project does

1. Simulates an industrial plant with 10 energy-consuming machines
   (motors, conveyor, compressor, pump, HVAC, boiler, cooling system,
   production line, lighting).
2. Generates a realistic 60,000+ row synthetic dataset with daily/weekly
   patterns, seasonal ambient temperature, machine degradation, and
   injected fault events.
3. Runs a live, in-memory **Digital Twin** that mirrors the same physics
   to simulate the plant in real time.
4. Trains and compares multiple **ML models** (Linear Regression, Random
   Forest, Gradient Boosting, optionally XGBoost) to predict energy
   consumption, and automatically selects the best one.
5. Forecasts future plant energy (1h / 6h / 24h / 7 days).
6. Detects anomalies (Isolation Forest + rule-based reasoning) with
   severity levels and probable causes.
7. Predicts machine health / maintenance risk.
8. **Optimizes** machine load allocation (SciPy constrained optimization)
   to reduce energy while meeting production targets.
9. Runs **what-if scenarios** (high/low production, peak tariff,
   degradation, equipment failure, energy-saving mode).
10. Calculates energy cost (flat / time-of-use tariff) and CO₂ emissions
    (configurable emission factor).
11. Presents everything in a **Streamlit dashboard** with a virtual plant
    layout, and generates downloadable **PDF / Excel reports**.

## 2. Project structure

```
industrial_energy_digital_twin/
├── app.py                     # Streamlit dashboard (entry point)
├── config.py                  # Central configuration
├── requirements.txt
├── README.md
│
├── data/
│   ├── industrial_energy_data.csv   # generated synthetic dataset
│   ├── machine_data.csv             # machine catalogue
│   └── digital_twin.db              # SQLite (created at runtime)
│
├── models/                    # trained model artifacts (.pkl, metrics.json)
│
├── src/
│   ├── data_generator.py      # synthetic dataset generator
│   ├── digital_twin.py        # live Digital Twin engine (Machine, Plant)
│   ├── train_energy_model.py  # ML model training + selection
│   ├── energy_forecasting.py  # time-series forecasting
│   ├── anomaly_detection.py   # Isolation Forest + rule-based reasons
│   ├── predictive_maintenance.py
│   ├── energy_optimizer.py    # SciPy constrained load optimization
│   ├── what_if_simulator.py   # scenario engine on top of the ML model
│   ├── energy_cost.py         # flat / time-of-use tariff cost calc
│   ├── carbon_calculator.py   # CO2 emission calc
│   ├── database.py            # SQLite persistence layer
│   └── report_generator.py    # PDF (ReportLab) + Excel reports
│
├── reports/                   # generated PDF/Excel reports land here
├── outputs/                   # misc exports
└── tests/                     # pytest unit tests
```

## 3. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Run

```bash
streamlit run app.py
```

The first launch will automatically:
- generate the synthetic dataset (`data/industrial_energy_data.csv`) if
  missing,
- train all AI models (energy prediction, forecasting, anomaly detection,
  predictive maintenance) if missing.

You can also run each step manually:

```bash
python -m src.data_generator          # generate/refresh dataset
python -m src.train_energy_model       # train + compare energy models
python -m src.energy_forecasting       # train forecasting model
python -m src.anomaly_detection        # train anomaly detector
python -m src.predictive_maintenance   # train maintenance classifier
```

## 5. Testing

```bash
pytest tests/ -v
```

## 6. Notes for evaluators / academic use

- Emission factors and tariffs are **configurable assumptions**, adjustable
  from the dashboard sidebar — they are not claims about a real utility.
- No cloud services or API keys are required; everything runs locally.
- The dataset is regenerable at any time from the "Generate New Dataset"
  button (or CLI), and the models can be retrained from "Retrain AI Model".
- `models/model_metrics.json` records the full comparison table (MAE, MSE,
  RMSE, R², MAPE) across all trained candidate models for transparency.
