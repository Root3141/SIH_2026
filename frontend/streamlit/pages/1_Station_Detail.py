import streamlit as st

from utils.styles import inject_css, COLOR_MAP, STATUS_ICON, STATUS_LABEL
from utils.explainability import render_explanation

# station_id -> station name, kept for reference / backend matching
STATION_IDS = {
    "AWS_001": "Delhi Central",
    "AWS_002": "Gurugram",
    "AWS_003": "Noida",
    "AWS_004": "Faridabad",
    "AWS_005": "Ghaziabad",
    "AWS_006": "Sonipat",
    "AWS_007": "Rohtak",
    "AWS_008": "Meerut",
    "AWS_009": "Jhajjar",
    "AWS_010": "Greater Noida",
    "AWS_011": "Bahadurgarh",
    "AWS_012": "Hapur",
}


st.set_page_config(page_title="Station Detail", page_icon="📡", layout="wide")
inject_css()

if "datasets" not in st.session_state or "time_index" not in st.session_state:
    st.warning("Please start from the main Dashboard page.")
    st.stop()

station = st.session_state.get("selected_station")
if not station:
    st.warning("No station selected. Go back to the dashboard and click a station.")
    if st.button("⬅️ Back to Dashboard"):
        st.switch_page("app.py")
    st.stop()

if st.button("⬅️ Back to Dashboard"):
    st.switch_page("app.py")

idx = st.session_state.time_index
df = st.session_state.datasets[station]
history = df.iloc[: idx + 1]
current = df.iloc[idx]
N_POINTS = len(df)

readings = {
    "Temperature": current["temperature"],
    "Pressure": current["pressure"],
    "Humidity": current["humidity"],
}
severity = current["final_severity"]
overall = {
    "normal": "green",
    "suspicious": "yellow",
    "anomaly": "red",
    "critical": "red",
}[severity]
detector_alerts = {
    "Statistical": bool(current["statistical_alert"]),
    "Spatial": bool(current["spatial_alert"]),
    "LSTM": bool(current["lstm_ae_alert"]),
}
explanations = [
    f"{detector} detector agreed"
    for detector, alert in detector_alerts.items()
    if alert
]
sensor_status = {sensor: (overall, f"Final status: {severity}") for sensor in readings}

station_name = STATION_IDS.get(station, station)

st.markdown(f"# 📡 {station_name}")
st.caption(
    f"Last reading: {current['timestamp'].strftime('%Y-%m-%d %H:%M')} (2026 stream, step {idx + 1}/{N_POINTS})"
)

st.markdown(
    f"""<div class="status-box" style="background-color:{COLOR_MAP[overall]}33;
    border:2px solid {COLOR_MAP[overall]}; font-size:24px;">
    {STATUS_ICON[overall]} {STATUS_LABEL[overall]}</div>""",
    unsafe_allow_html=True,
)

st.markdown("---")

col1, col2, col3 = st.columns(3)
for col, sensor in zip([col1, col2, col3], ["Temperature", "Pressure", "Humidity"]):
    s_status, s_msg = sensor_status[sensor]
    col.metric(sensor, f"{readings[sensor]:.1f}", help=s_msg)
    col.markdown(
        f"<span style='color:{COLOR_MAP[s_status]};font-weight:600;'>{STATUS_LABEL[s_status].upper()}</span>",
        unsafe_allow_html=True,
    )

st.markdown("---")

render_explanation(current)

st.markdown("### 📈 Live Sensor Trends")
tabs = st.tabs(["Temperature", "Pressure", "Humidity"])
for tab, sensor in zip(tabs, ["Temperature", "Pressure", "Humidity"]):
    with tab:
        chart_df = history.set_index("timestamp")[[sensor.lower()]]
        chart_df.columns = [sensor]
        st.line_chart(chart_df)
