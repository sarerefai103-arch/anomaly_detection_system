"""
dashboard.py
-------------
المرحلة 5: الداشبورد التفاعلي لنظام كشف الانحرافات.
ثيم: أسود + برتقالي، مع انيميشنز CSS ورسوم بيانية متحركة (Plotly).

للتشغيل محلياً (مجاني بالكامل):
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402

from app_config import (  # noqa: E402
    AN_TAB_ANOMALY,
    AN_TAB_BUSINESS,
    AN_TAB_DRIFT,
    AN_TAB_PREDICTIVE,
    DEVELOPER,
    NAV_ANALYTICS,
    NAV_CAMPAIGN,
    NAV_CHAT,
    NAV_HOME,
)
from services import resolve_business_source  # noqa: E402
from sidebar import sidebar_controls  # noqa: E402
from ui import load_css, render_data_strip  # noqa: E402
from views.anomaly import render_anomaly_tab  # noqa: E402
from views.business import render_business_tab  # noqa: E402
from views.campaigns import render_campaign_tab  # noqa: E402
from views.chat import render_chat_tab  # noqa: E402
from views.drift import render_drift_tab  # noqa: E402
from views.home import render_home_tab  # noqa: E402
from views.predictive import render_predictive_tab  # noqa: E402

st.set_page_config(
    page_title="نظام كشف الانحرافات | Anomaly Detection System",
    page_icon="🟠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_analytics_section(market, ai_conf, source, algo, contamination, run_btn):
    render_data_strip(source)
    st.write("")

    tab_anomaly, tab_business, tab_predictive, tab_drift = st.tabs(
        [AN_TAB_ANOMALY, AN_TAB_BUSINESS, AN_TAB_PREDICTIVE, AN_TAB_DRIFT],
        key="analytics_tabs",
    )

    with tab_anomaly:
        render_anomaly_tab(source, algo, contamination, run_btn)

    with tab_business:
        render_business_tab(market, ai_conf, source)

    with tab_predictive:
        render_predictive_tab(market, ai_conf, source)

    with tab_drift:
        render_drift_tab(source)


def main():
    load_css()

    st.markdown(
        f"""
        <div class="fade-block">
            <div class="hero-title">نظام تحليل البيانات الذكي</div>
            <div class="hero-subtitle">
                Smart Data Analytics — كشف الانحرافات، تحليلات الأعمال، التحليل التنبؤي، مراقبة تغيّر البيانات، وتخطيط الحملات
            </div>
            <div class="hero-dev">👨‍💻 المطوّر | Developer: <strong>{DEVELOPER}</strong></div>
        </div>
        <br>
        """,
        unsafe_allow_html=True,
    )

    (
        uploaded, db_source, data_mode,
        algo, contamination, run_btn, market, ai_conf,
    ) = sidebar_controls()
    biz_source = resolve_business_source(uploaded, db_source, data_mode)

    # التنقل الرئيسي: الرئيسية + 3 أقسام، مع إمكانية الانتقال برمجيًا من البطاقات
    st.session_state.setdefault("main_tabs", NAV_HOME)
    tab_home, tab_analytics, tab_campaign, tab_chat = st.tabs(
        [NAV_HOME, NAV_ANALYTICS, NAV_CAMPAIGN, NAV_CHAT],
        key="main_tabs",
        on_change="rerun",
    )

    with tab_home:
        render_home_tab(biz_source, market)

    with tab_analytics:
        render_analytics_section(market, ai_conf, biz_source, algo, contamination, run_btn)

    with tab_campaign:
        render_campaign_tab(market)

    with tab_chat:
        render_data_strip(biz_source)
        st.write("")
        render_chat_tab(market, ai_conf, biz_source)


if __name__ == "__main__":
    main()
