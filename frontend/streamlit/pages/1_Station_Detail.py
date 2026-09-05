import streamlit as st

from utils.simulator import N_POINTS
from utils.detector import evaluate_station
from utils.styles import inject_css, COLOR_MAP, STATUS_ICON, STATUS_LABEL

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

readings = {
    "Temperature": current["Temperature"],
    "Pressure": current["Pressure"],
    "Humidity": current["Humidity"],
}
overall, explanations, per_sensor = evaluate_station(readings)

st.markdown(f"# 📡 {station}")
st.caption(f"Last reading: {current['timestamp'].strftime('%Y-%m-%d %H:%M')} (2026 stream, step {idx + 1}/{N_POINTS})")

st.markdown(
    f"""<div class="status-box" style="background-color:{COLOR_MAP[overall]}33;
    border:2px solid {COLOR_MAP[overall]}; font-size:24px;">
    {STATUS_ICON[overall]} {STATUS_LABEL[overall]}</div>""",
    unsafe_allow_html=True,
)

st.markdown("---")

col1, col2, col3 = st.columns(3)
for col, sensor in zip([col1, col2, col3], ["Temperature", "Pressure", "Humidity"]):
    s_status, s_msg = per_sensor[sensor]
    col.metric(sensor, f"{readings[sensor]:.1f}", help=s_msg)
    col.markdown(
        f"<span style='color:{COLOR_MAP[s_status]};font-weight:600;'>{s_status.upper()}</span>",
        unsafe_allow_html=True,
    )

st.markdown("---")

if explanations:
    st.markdown("### ⚠️ Alert Explanations")
    for e in explanations:
        st.warning(e)
else:
    st.success("All sensors within normal range.")

st.markdown("### 📈 Live Sensor Trends")
tabs = st.tabs(["Temperature", "Pressure", "Humidity"])
for tab, sensor in zip(tabs, ["Temperature", "Pressure", "Humidity"]):
    with tab:
        chart_df = history.set_index("timestamp")[[sensor]]
        st.line_chart(chart_df)
