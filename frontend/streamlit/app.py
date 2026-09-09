import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from pathlib import Path

from utils.styles import inject_css, COLOR_MAP, STATUS_ICON

RESULTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "prototype"
    / "skyguard_demo_2026_results.parquet"
)


@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Precomputed SkyGuard results not found: {RESULTS_PATH}"
        )

    results = pd.read_parquet(RESULTS_PATH)
    results["timestamp"] = pd.to_datetime(results["timestamp"], utc=True)
    return results.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


st.set_page_config(
    page_title="Anomaly Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ---------------- SESSION STATE ----------------
results = load_results()
if "results" not in st.session_state:
    st.session_state.results = results
if "datasets" not in st.session_state:
    st.session_state.datasets = {
        station_id: station_df.reset_index(drop=True)
        for station_id, station_df in results.groupby("station_id", sort=True)
    }
if "time_index" not in st.session_state:
    st.session_state.time_index = 0
if "streaming" not in st.session_state:
    st.session_state.streaming = True
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
if "selected_station" not in st.session_state:
    st.session_state.selected_station = None

STATIONS = {
    station_id: (
        float(station_df["latitude"].iloc[0]),
        float(station_df["longitude"].iloc[0]),
    )
    for station_id, station_df in st.session_state.datasets.items()
}
N_POINTS = min(len(station_df) for station_df in st.session_state.datasets.values())

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ Simulation Controls")
    st.markdown("---")

    interval_label = st.selectbox(
        "⏱️ Stream Speed", ["15 seconds", "30 seconds", "1 minute"], index=0
    )
    interval_seconds = {"15 seconds": 15, "30 seconds": 30, "1 minute": 60}[
        interval_label
    ]

    st.session_state.streaming = st.toggle(
        "▶️ Streaming", value=st.session_state.streaming
    )

    if st.button("🔁 Reset Stream", use_container_width=True):
        st.session_state.time_index = 0
        st.session_state.alert_log = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        f"**2026 dataset position:** {st.session_state.time_index + 1} / {N_POINTS}"
    )
    st.progress((st.session_state.time_index + 1) / N_POINTS)

    st.markdown("---")
    st.markdown("### 🗺️ Monitored Stations")
    for p in STATIONS:
        st.markdown(f"- {p}")

# ---------------- STREAM TICK ----------------
if st.session_state.streaming:
    st_autorefresh(interval=interval_seconds * 1000, key="stream_tick")
    st.session_state.time_index = min(st.session_state.time_index + 1, N_POINTS - 1)

# ---------------- COMPUTE CURRENT STATUS PER STATION ----------------
idx = st.session_state.time_index
station_status = {}
for name, df in st.session_state.datasets.items():
    row = df.iloc[idx]
    readings = {
        "Temperature": row["temperature"],
        "Pressure": row["pressure"],
        "Humidity": row["humidity"],
    }
    severity = row["final_severity"]
    overall = {
        "normal": "green",
        "suspicious": "yellow",
        "anomaly": "red",
        "critical": "red",
    }[severity]
    explanations = [
        f"{label} detector agreed"
        for label, column in (
            ("Statistical", "statistical_alert"),
            ("Spatial", "spatial_alert"),
            ("LSTM", "lstm_ae_alert"),
        )
        if bool(row[column])
    ]
    per_sensor = {}
    station_status[name] = {
        "overall": overall,
        "explanations": explanations,
        "readings": readings,
        "per_sensor": per_sensor,
        "timestamp": row["timestamp"],
        "severity": severity,
        "detector_votes": int(row["detector_votes"]),
    }
    if overall in ("yellow", "red"):
        last = st.session_state.alert_log[-1] if st.session_state.alert_log else None
        if not last or not (last["Station"] == name and last["idx"] == idx):
            st.session_state.alert_log.append(
                {
                    "Station": name,
                    "Status": overall,
                    "idx": idx,
                    "Time": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "Details": "; ".join(explanations),
                }
            )

# ---------------- HEADER ----------------
st.markdown(
    '<p class="main-title">🚨 Anomaly Detection System</p>', unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-title">SIH Demo — Real-time monitoring across 13 locations · streaming 2026 dataset</p>',
    unsafe_allow_html=True,
)

# ---------------- TOP METRICS ----------------
green_count = sum(1 for s in station_status.values() if s["overall"] == "green")
yellow_count = sum(1 for s in station_status.values() if s["overall"] == "yellow")
red_count = sum(1 for s in station_status.values() if s["overall"] == "red")

col1, col2, col3, col4 = st.columns(4)
col1.metric("🟢 Normal", green_count)
col2.metric("🟡 Suspicious", yellow_count)
col3.metric("🔴 Alerts", red_count)
col4.metric("📈 Total Alerts Logged", len(st.session_state.alert_log))

st.markdown("---")

tab1, tab2 = st.tabs(["🗺️ Live Map", "📋 Alert Log"])

with tab1:
    latest_ts = station_status[next(iter(STATIONS))]["timestamp"].strftime(
        "%Y-%m-%d %H:%M"
    )
    st.subheader(f"Live Status — {latest_ts}")

    map_df = pd.DataFrame(
        [
            {
                "Station": name,
                "lat": STATIONS[name][0],
                "lon": STATIONS[name][1],
                "Status": station_status[name]["overall"].capitalize(),
                "Temperature": round(
                    station_status[name]["readings"]["Temperature"], 1
                ),
                "Pressure": round(station_status[name]["readings"]["Pressure"], 1),
                "Humidity": round(station_status[name]["readings"]["Humidity"], 1),
            }
            for name in STATIONS
        ]
    )

    map_center = {"lat": map_df["lat"].mean(), "lon": map_df["lon"].mean()}

    map_kwargs = dict(
        lat="lat",
        lon="lon",
        color="Status",
        color_discrete_map={"Green": "#2ECC71", "Yellow": "#F1C40F", "Red": "#E74C3C"},
        hover_name="Station",
        hover_data={
            "Temperature": True,
            "Pressure": True,
            "Humidity": True,
            "lat": False,
            "lon": False,
        },
        center=map_center,
        zoom=8.7,
        height=520,
    )
    if hasattr(px, "scatter_map"):
        # plotly >= 5.24: scatter_mapbox was renamed to scatter_map (MapLibre, no token needed)
        fig = px.scatter_map(map_df, **map_kwargs)
        fig.update_layout(map_style="carto-darkmatter", margin=dict(l=0, r=0, t=0, b=0))
    else:
        fig = px.scatter_mapbox(map_df, **map_kwargs)
        fig.update_layout(
            mapbox_style="carto-darkmatter", margin=dict(l=0, r=0, t=0, b=0)
        )
    fig.update_traces(marker=dict(size=18))

    event = st.plotly_chart(
        fig, use_container_width=True, on_select="rerun", key="station_map"
    )

    clicked_station = None
    if event and event.get("selection") and event["selection"].get("points"):
        point = event["selection"]["points"][0]
        clicked_station = map_df.iloc[point["point_index"]]["Station"]

    st.caption(
        "Click a marker on the map, or a card below, to open that station's live detail page."
    )

    st.markdown("### Station Grid")
    cols = st.columns(4)
    for i, name in enumerate(STATIONS):
        status = station_status[name]["overall"]
        with cols[i % 4]:
            st.markdown(
                f"""<div class="station-card" style="background-color:{COLOR_MAP[status]}22;
                border:2px solid {COLOR_MAP[status]};">
                {STATUS_ICON[status]} {name}</div>""",
                unsafe_allow_html=True,
            )
            if st.button("Open", key=f"open_{name}", use_container_width=True):
                clicked_station = name

    if clicked_station:
        st.session_state.selected_station = clicked_station
        st.switch_page("pages/1_Station_Detail.py")

with tab2:
    st.subheader("📋 Alert Log")
    if st.session_state.alert_log:
        log_df = pd.DataFrame(st.session_state.alert_log)[
            ["Time", "Station", "Status", "Details"]
        ]
        st.dataframe(log_df.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No alerts yet — waiting for the stream.")
