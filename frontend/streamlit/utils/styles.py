import streamlit as st

COLOR_MAP = {"green": "#2ECC71", "yellow": "#F1C40F", "red": "#E74C3C"}

STATUS_ICON = {"green": "🟢", "yellow": "🟡", "red": "🔴"}

STATUS_LABEL = {"green": "SYSTEM NORMAL", "yellow": "SUSPICIOUS", "red": "ALERT"}


def inject_css():
    """Injects the shared dark dashboard theme. Call once at the top of every page."""
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 42px;
            font-weight: 800;
            color: #FF4B4B;
            text-align: center;
            margin-bottom: 0px;
        }
        .sub-title {
            text-align: center;
            color: #AAAAAA;
            font-size: 16px;
            margin-bottom: 30px;
        }
        div[data-testid="stMetric"] {
            background-color: #1E1E1E;
            border: 1px solid #333;
            padding: 15px;
            border-radius: 12px;
        }
        .status-box {
            padding: 15px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            text-align: center;
        }
        .station-card {
            padding: 10px;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
