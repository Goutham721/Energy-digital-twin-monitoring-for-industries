"""
app.py

Main Streamlit application: a professional industrial energy-management
dashboard built entirely on top of the software-simulated Digital Twin.
No hardware, sensors, or IoT devices are involved -- see config.DISCLAIMER.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.anomaly_detection import (detect_anomalies, get_recent_anomalies,
                                    train_anomaly_model)
from src.carbon_calculator import calculate_emissions, emissions_summary
from src.data_generator import generate_dataset, load_dataset
from src.database import Database
from src.digital_twin import IndustrialPlantDigitalTwin
from src.energy_cost import calculate_cost, cost_summary
from src.energy_forecasting import forecast_future, train_forecast_model
from src.energy_optimizer import optimize_plant_load
from src.predictive_maintenance import (assess_fleet_health,
                                         train_maintenance_model)
from src.report_generator import generate_excel_report, generate_pdf_report
from src.train_energy_model import (get_feature_importance, load_trained_model,
                                     predict_energy, train_and_select_best)
from src.what_if_simulator import PRESET_SCENARIOS, run_preset, run_scenario

# ---------------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    .stApp { background-color: #0e1420; color: #e6e9ef; }
    section[data-testid="stSidebar"] { background-color: #131c2e; }
    div[data-testid="stMetric"] {
        background-color: #182338; border: 1px solid #24314a;
        padding: 12px; border-radius: 10px;
    }
    h1, h2, h3 { color: #e6e9ef; }
    .disclaimer-box {
        background-color: #1c2740; border-left: 4px solid #4c7cf0;
        padding: 10px 14px; border-radius: 6px; font-size: 0.85rem; color: #b7c2d6;
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SESSION STATE / CACHED RESOURCES
# ---------------------------------------------------------------------------
@st.cache_resource
def get_db() -> Database:
    return Database()


@st.cache_data(show_spinner=False)
def get_dataset(_version: int = 0) -> pd.DataFrame:
    """`_version` busts the cache after regenerating data."""
    return load_dataset()


def get_twin() -> IndustrialPlantDigitalTwin:
    if "twin" not in st.session_state:
        st.session_state.twin = IndustrialPlantDigitalTwin()
    return st.session_state.twin


def ensure_models_trained() -> None:
    """Best-effort auto-train of any missing model so pages don't crash on first run."""
    if not config.BEST_MODEL_PATH.exists():
        with st.spinner("Training AI energy-prediction model for the first time..."):
            train_and_select_best()
    if not config.ANOMALY_MODEL_PATH.exists():
        with st.spinner("Training anomaly detection model..."):
            train_anomaly_model()
    if not config.MAINTENANCE_MODEL_PATH.exists():
        with st.spinner("Training predictive maintenance model..."):
            train_maintenance_model()
    if not config.FORECAST_MODEL_PATH.exists():
        with st.spinner("Training forecasting model..."):
            train_forecast_model()


if "data_version" not in st.session_state:
    st.session_state.data_version = 0

if not config.DATASET_PATH.exists():
    with st.spinner("Generating initial synthetic industrial dataset (first run only)..."):
        generate_dataset(num_records=config.DEFAULT_NUM_RECORDS, save=True)

db = get_db()

# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
st.sidebar.title("⚡ Digital Twin Control")
st.sidebar.markdown(f"<div class='disclaimer-box'>{config.DISCLAIMER}</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

PAGES = [
    "Dashboard", "Digital Twin", "Energy Monitoring", "AI Prediction",
    "Forecasting", "Anomaly Detection", "Predictive Maintenance",
    "Optimization", "What-If Simulation", "Reports",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.subheader("Demo Controls")

if st.sidebar.button("▶ Run Digital Twin Simulation Step"):
    twin = get_twin()
    state = twin.simulate_step()
    db.log_simulation_snapshot(state)
    st.sidebar.success(f"Advanced to {state['timestamp']}")

if st.sidebar.button("🔄 Generate New Dataset"):
    with st.spinner(f"Generating {config.DEFAULT_NUM_RECORDS:,} synthetic records..."):
        generate_dataset(num_records=config.DEFAULT_NUM_RECORDS, save=True)
        st.session_state.data_version += 1
        get_dataset.clear()
    st.sidebar.success("New dataset generated.")

if st.sidebar.button("🤖 Retrain AI Model"):
    with st.spinner("Retraining energy prediction model..."):
        result = train_and_select_best()
    st.sidebar.success(f"Best model: {result['best_model_name']} (Test R²={result['test_metrics']['R2']})")

st.sidebar.markdown("---")
tariff_mode = st.sidebar.selectbox("Tariff Mode", ["flat", "time_of_use"], index=0)
custom_tariff = st.sidebar.number_input(
    "Flat Tariff ($/kWh)", min_value=0.01, max_value=2.0,
    value=config.DEFAULT_FLAT_TARIFF_PER_KWH, step=0.01,
)
emission_factor = st.sidebar.number_input(
    "Emission Factor (kg CO2/kWh)", min_value=0.0, max_value=2.0,
    value=config.DEFAULT_EMISSION_FACTOR_KG_PER_KWH, step=0.01,
)

ensure_models_trained()
df = get_dataset(st.session_state.data_version)
twin = get_twin()

# ===========================================================================
# PAGE: DASHBOARD
# ===========================================================================
if page == "Dashboard":
    st.title(config.APP_TITLE)
    st.caption(config.APP_SUBTITLE)

    state = twin.get_plant_state()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Energy (kWh, this step)", f"{state['total_energy_kwh']:.2f}")
    c2.metric("Current Power (kW)", f"{state['total_power_kw']:.2f}")
    c3.metric("Energy Cost", f"${state['energy_cost']:.2f}")
    c4.metric("CO₂ Emissions (kg)", f"{state['carbon_emission_kg']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Production Target", f"{state['production_target']:.0f}%")
    c6.metric("Average Efficiency", f"{state['average_efficiency'] * 100:.1f}%")
    c7.metric("Active Machines", f"{state['active_machines']} / {state['total_machines']}")
    c8.metric("Active Anomalies", state["active_anomalies"])

    st.markdown("### Historical Energy Overview (Synthetic Dataset)")
    daily = df.copy()
    daily["date"] = pd.to_datetime(daily["timestamp"]).dt.date
    daily_energy = daily.groupby("date")["energy_consumption_kwh"].sum().reset_index()
    fig = px.line(daily_energy, x="date", y="energy_consumption_kwh",
                   title="Total Daily Plant Energy Consumption (kWh)")
    fig.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        by_type = df.groupby("machine_type")["energy_consumption_kwh"].sum().reset_index()
        fig2 = px.pie(by_type, names="machine_type", values="energy_consumption_kwh",
                       title="Energy Share by Machine Type", hole=0.4)
        fig2.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        eff_by_type = df.groupby("machine_type")["efficiency"].mean().reset_index()
        fig3 = px.bar(eff_by_type, x="machine_type", y="efficiency",
                       title="Average Efficiency by Machine Type")
        fig3.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig3, use_container_width=True)

# ===========================================================================
# PAGE: DIGITAL TWIN
# ===========================================================================
elif page == "Digital Twin":
    st.title("🏭 Virtual Plant — Digital Twin")
    st.caption("Live, in-memory simulation of the industrial plant. Click 'Run Digital Twin "
               "Simulation Step' in the sidebar to advance the simulation clock.")

    state = twin.get_plant_state()
    st.info(f"Simulation time: **{state['timestamp']}**  |  Production target: **{state['production_target']:.0f}%**")

    layout = {
        "Production Line": ["Motor_1", "Motor_2", "Conveyor_1", "Production_Line_1"],
        "Utilities": ["Compressor_1", "Pump_1", "Cooling_System_1"],
        "Building Systems": ["HVAC_1", "Boiler_1", "Lighting_1"],
    }

    status_color = {"Running": "🟢", "Idle": "🟡", "Stopped": "🔴"}
    health_color = lambda h: "🟢" if h >= 75 else ("🟡" if h >= 50 else "🔴")

    for zone, machine_ids in layout.items():
        st.markdown(f"#### {zone}")
        cols = st.columns(len(machine_ids))
        for col, mid in zip(cols, machine_ids):
            m = state["machines"].get(mid)
            if not m:
                continue
            with col:
                st.markdown(f"**{status_color.get(m['status'], '⚪')} {mid}**")
                st.caption(m["machine_type"])
                st.write(f"Status: `{m['status']}`")
                st.write(f"Load: {m['load_percentage']:.1f}%")
                st.write(f"Power: {m['power_kw']:.2f} kW")
                st.write(f"Efficiency: {m['efficiency'] * 100:.1f}%")
                st.write(f"Health: {health_color(m['health_score'])} {m['health_score']}")
                if m["anomaly_flag"]:
                    st.error(f"⚠ {m['fault_type']}")

    st.markdown("---")
    st.markdown("### Machine Detail Table")
    mdf = pd.DataFrame(list(state["machines"].values()))
    st.dataframe(mdf, use_container_width=True, hide_index=True)

# ===========================================================================
# PAGE: ENERGY MONITORING
# ===========================================================================
elif page == "Energy Monitoring":
    st.title("📊 Energy Monitoring")

    df_cost = calculate_cost(df, tariff_mode=tariff_mode, flat_rate=custom_tariff)
    summary = cost_summary(df_cost)
    c1, c2, c3 = st.columns(3)
    c1.metric("Daily Avg Cost", f"${summary['daily_avg_cost']:.2f}")
    c2.metric("Monthly Estimated Cost", f"${summary['monthly_estimated_cost']:,.2f}")
    c3.metric("Total Cost (dataset window)", f"${summary['total_cost']:,.2f}")

    df_ts = df.copy()
    df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])
    hourly = df_ts.set_index("timestamp").resample("1h")["energy_consumption_kwh"].sum().reset_index()
    fig = px.line(hourly, x="timestamp", y="energy_consumption_kwh", title="Hourly Plant Energy (kWh)")
    fig.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        by_machine = df.groupby("machine_id")["energy_consumption_kwh"].sum().sort_values(ascending=False).reset_index()
        fig2 = px.bar(by_machine, x="machine_id", y="energy_consumption_kwh", title="Machine-wise Total Energy (kWh)")
        fig2.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        df_ts["hour"] = df_ts["timestamp"].dt.hour
        by_hour = df_ts.groupby("hour")["power_kw"].mean().reset_index()
        peak_hour = by_hour.loc[by_hour["power_kw"].idxmax(), "hour"]
        fig3 = px.bar(by_hour, x="hour", y="power_kw", title=f"Avg Power by Hour (Peak ≈ {int(peak_hour)}:00)")
        fig3.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig3, use_container_width=True)

# ===========================================================================
# PAGE: AI PREDICTION
# ===========================================================================
elif page == "AI Prediction":
    st.title("🤖 AI Energy Prediction")
    st.caption("Predict energy consumption (kWh) for a given operating condition using the trained model.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            machine_type = st.selectbox("Machine Type", sorted(df["machine_type"].unique()))
            machine_status = st.selectbox("Machine Status", ["Running", "Idle", "Stopped"])
            maintenance_status = st.selectbox("Maintenance Status", ["Good", "Needs_Attention", "Critical"])
        with col2:
            load_percentage = st.slider("Load (%)", 0.0, 130.0, 65.0)
            production_rate = st.slider("Production Rate (%)", 0.0, 100.0, 70.0)
            efficiency = st.slider("Efficiency", 0.4, 0.99, 0.88)
        with col3:
            temperature = st.slider("Temperature (°C)", 0.0, 90.0, 35.0)
            pressure = st.slider("Pressure (bar)", 0.5, 3.0, 1.3)
            vibration = st.slider("Vibration", 0.0, 5.0, 0.8)

        col4, col5, col6 = st.columns(3)
        with col4:
            operating_hours = st.number_input("Operating Hours", 0, 300000, 5000)
        with col5:
            ambient_temperature = st.slider("Ambient Temperature (°C)", -10.0, 45.0, 24.0)
        with col6:
            hour = st.slider("Hour of Day", 0, 23, 14)

        submitted = st.form_submit_button("Predict Energy Consumption")

    if submitted:
        day_of_week = pd.Timestamp.now().dayofweek
        input_dict = {
            "machine_type": machine_type, "machine_status": machine_status,
            "maintenance_status": maintenance_status, "load_percentage": load_percentage,
            "production_rate": production_rate, "efficiency": efficiency,
            "temperature": temperature, "pressure": pressure, "vibration": vibration,
            "operating_hours": operating_hours, "ambient_temperature": ambient_temperature,
            "hour": hour, "day_of_week": day_of_week, "is_weekend": int(day_of_week >= 5),
        }
        pred = predict_energy(input_dict)
        db.log_prediction(input_dict, pred)

        st.success(f"### Predicted Energy Consumption: **{pred:.2f} kWh**")
        c1, c2 = st.columns(2)
        c1.metric("Estimated Cost", f"${pred * custom_tariff:.2f}")
        c2.metric("Estimated CO₂", f"{pred * emission_factor:.2f} kg")

    st.markdown("---")
    st.markdown("### Model Explainability — Feature Importance")
    fi = get_feature_importance()
    if not fi.empty:
        fig = px.bar(fi, x="importance", y="feature", orientation="h",
                      title="Top Factors Driving Energy Consumption")
        fig.update_layout(template="plotly_dark", height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance is only available for tree-based models.")

# ===========================================================================
# PAGE: FORECASTING
# ===========================================================================
elif page == "Forecasting":
    st.title("📈 Energy Forecasting")
    horizon = st.selectbox("Forecast Horizon", ["Next Hour", "Next 6 Hours", "Next 24 Hours", "Next 7 Days"], index=2)
    horizon_map = {"Next Hour": 1, "Next 6 Hours": 6, "Next 24 Hours": 24, "Next 7 Days": 168}

    with st.spinner("Generating forecast..."):
        forecast_df = forecast_future(horizon_hours=horizon_map[horizon])

    fig = px.line(forecast_df, x="timestamp", y="predicted_energy_kwh",
                   title=f"Forecasted Total Plant Energy — {horizon}")
    fig.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig, use_container_width=True)

    peak_row = forecast_df.loc[forecast_df["predicted_energy_kwh"].idxmax()]
    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast Avg (kWh/step)", f"{forecast_df['predicted_energy_kwh'].mean():.2f}")
    c2.metric("Peak Predicted (kWh)", f"{peak_row['predicted_energy_kwh']:.2f}")
    c3.metric("Peak Period", str(peak_row["timestamp"]))

    st.markdown("### Forecast Data")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

# ===========================================================================
# PAGE: ANOMALY DETECTION
# ===========================================================================
elif page == "Anomaly Detection":
    st.title("🚨 Anomaly Detection")

    with st.spinner("Scanning for anomalies..."):
        anomalies = get_recent_anomalies(df, top_n=200)

    db.log_anomalies(anomalies)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Anomalies Flagged", len(anomalies))
    c2.metric("Critical", int((anomalies["severity"] == "CRITICAL").sum()) if len(anomalies) else 0)
    c3.metric("Warning", int((anomalies["severity"] == "WARNING").sum()) if len(anomalies) else 0)

    if not anomalies.empty:
        fig = px.scatter(
            anomalies, x="timestamp", y="anomaly_score", color="severity",
            hover_data=["machine_id", "probable_reason"],
            title="Anomaly Score Over Time",
            color_discrete_map={"CRITICAL": "#e63946", "WARNING": "#f4a261"},
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

        by_machine = anomalies["machine_id"].value_counts().reset_index()
        by_machine.columns = ["machine_id", "anomaly_count"]
        fig2 = px.bar(by_machine, x="machine_id", y="anomaly_count", title="Anomalies by Machine")
        fig2.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Anomaly Log")
        st.dataframe(anomalies, use_container_width=True, hide_index=True)
    else:
        st.success("No anomalies detected in the current dataset window.")

# ===========================================================================
# PAGE: PREDICTIVE MAINTENANCE
# ===========================================================================
elif page == "Predictive Maintenance":
    st.title("🔧 Predictive Maintenance")

    with st.spinner("Assessing machine health..."):
        health = assess_fleet_health(df)

    db.log_maintenance_events(health)

    condition_colors = {
        "Healthy": "#2a9d8f", "Warning": "#f4a261",
        "Maintenance_Required": "#e76f51", "Critical": "#e63946",
    }

    cols = st.columns(len(health))
    for col, (_, row) in zip(cols, health.iterrows()):
        with col:
            color = condition_colors.get(row["predicted_condition"], "#999")
            st.markdown(f"**{row['machine_id']}**")
            st.markdown(f"<span style='color:{color}; font-weight:bold'>{row['predicted_condition']}</span>",
                        unsafe_allow_html=True)
            st.write(f"Risk Score: {row['risk_score']:.0f}/100")

    st.markdown("---")
    fig = px.bar(health.sort_values("risk_score", ascending=True), x="risk_score", y="machine_id",
                  orientation="h", color="predicted_condition",
                  color_discrete_map=condition_colors, title="Machine Risk Score")
    fig.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Maintenance Recommendations")
    st.dataframe(health, use_container_width=True, hide_index=True)

# ===========================================================================
# PAGE: OPTIMIZATION
# ===========================================================================
elif page == "Optimization":
    st.title("⚙️ Energy Optimization")
    st.caption("Recommends machine load re-allocation to reduce energy while meeting production needs.")

    state = twin.get_plant_state()
    machine_states = list(state["machines"].values())

    required_prod = st.slider("Required Production Level (%)", 30, 100, 75)

    if st.button("Run Optimization"):
        with st.spinner("Solving constrained load-optimization problem..."):
            result = optimize_plant_load(
                machine_states, required_production=required_prod,
                tariff_per_kwh=custom_tariff, emission_factor=emission_factor,
            )
        db.log_optimization_run(result)
        st.session_state["last_optimization"] = result

    result = st.session_state.get("last_optimization")
    if result:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Energy", f"{result['current_energy_kwh']:.2f} kWh")
        c2.metric("Optimized Energy", f"{result['optimized_energy_kwh']:.2f} kWh")
        c3.metric("Energy Saved", f"{result['energy_saved_kwh']:.2f} kWh")
        c4.metric("Savings", f"{result['savings_percent']:.2f}%")

        c5, c6 = st.columns(2)
        c5.metric("Cost Savings", f"${result['cost']['cost_savings']:.2f}")
        c6.metric("CO₂ Reduction", f"{result['co2']['emission_reduction_kg']:.2f} kg")

        rec_df = pd.DataFrame(result["recommendations"])
        if not rec_df.empty:
            fig = go.Figure()
            fig.add_bar(name="Current Load %", x=rec_df["machine_id"], y=rec_df["current_load_percent"])
            fig.add_bar(name="Recommended Load %", x=rec_df["machine_id"], y=rec_df["recommended_load_percent"])
            fig.update_layout(barmode="group", template="plotly_dark", height=400,
                               title="Current vs Recommended Machine Load")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Run Optimization' to generate recommendations for the current Digital Twin state.")

# ===========================================================================
# PAGE: WHAT-IF SIMULATION
# ===========================================================================
elif page == "What-If Simulation":
    st.title("🔮 What-If Simulation")

    baseline_machine = st.selectbox("Baseline Machine", sorted(df["machine_id"].unique()))
    baseline_row = df[df["machine_id"] == baseline_machine].sort_values("timestamp").iloc[-1]
    baseline = {c: baseline_row[c] for c in config.ALL_FEATURES if c in baseline_row}

    st.markdown("#### Baseline Condition")
    st.json({k: (round(v, 3) if isinstance(v, float) else v) for k, v in baseline.items()})

    preset = st.selectbox("Scenario Preset", list(PRESET_SCENARIOS.keys()))

    st.markdown("#### Or customize manually")
    col1, col2 = st.columns(2)
    with col1:
        load_delta = st.slider("Load % change", -50, 50, 0)
        prod_delta = st.slider("Production rate % change", -50, 50, 0)
    with col2:
        eff_delta = st.slider("Efficiency change", -0.3, 0.3, 0.0, step=0.01)
        temp_delta = st.slider("Temperature change (°C)", -20, 40, 0)

    manual_overrides = {
        "load_percentage_delta": load_delta, "production_rate_delta": prod_delta,
        "efficiency_delta": eff_delta, "temperature_delta": temp_delta,
    }
    manual_overrides = {k: v for k, v in manual_overrides.items() if v}

    if st.button("Run What-If Scenario"):
        overrides = {**PRESET_SCENARIOS.get(preset, {}), **manual_overrides}
        result = run_scenario(baseline, overrides, tariff_per_kwh=custom_tariff, emission_factor=emission_factor)

        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline Energy", f"{result['baseline']['energy_kwh']:.2f} kWh")
        c2.metric("Scenario Energy", f"{result['scenario']['energy_kwh']:.2f} kWh",
                   delta=f"{result['energy_delta_kwh']:.2f} kWh")
        c3.metric("Change", f"{result['energy_delta_percent']:.1f}%")

        c4, c5 = st.columns(2)
        c4.metric("Cost Change", f"${result['cost_delta']:.2f}")
        c5.metric("CO₂ Change", f"{result['co2_delta_kg']:.2f} kg")

        fig = go.Figure(data=[
            go.Bar(name="Baseline", x=["Energy (kWh)"], y=[result["baseline"]["energy_kwh"]]),
            go.Bar(name="Scenario", x=["Energy (kWh)"], y=[result["scenario"]["energy_kwh"]]),
        ])
        fig.update_layout(barmode="group", template="plotly_dark", height=350, title="Baseline vs Scenario")
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# PAGE: REPORTS
# ===========================================================================
elif page == "Reports":
    st.title("📄 Reports")
    st.caption("Generate downloadable PDF / Excel reports summarizing the current plant state.")

    state = twin.get_plant_state()
    anomalies = get_recent_anomalies(df, top_n=20)
    health = assess_fleet_health(df)
    optimization_result = st.session_state.get("last_optimization")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate PDF Report"):
            with st.spinner("Building PDF report..."):
                path = generate_pdf_report(state, anomalies, health, optimization_result)
            with open(path, "rb") as f:
                st.download_button("⬇ Download PDF Report", f, file_name=Path(path).name, mime="application/pdf")
    with col2:
        if st.button("Generate Excel Report"):
            with st.spinner("Building Excel report..."):
                path = generate_excel_report(state, anomalies, health)
            with open(path, "rb") as f:
                st.download_button(
                    "⬇ Download Excel Report", f, file_name=Path(path).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    st.markdown("---")
    st.markdown("### Raw Dataset Download")
    st.download_button(
        "⬇ Download Full Dataset (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="industrial_energy_data.csv", mime="text/csv",
    )
