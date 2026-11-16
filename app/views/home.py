"""Landing page with navigation cards."""
from __future__ import annotations

import streamlit as st

from app_config import NAV_ANALYTICS, NAV_CAMPAIGN, NAV_CHAT
from services import _quality_cached, _read_cached
from ui import group_label, kpi_card


def _go_main(label: str) -> None:
    """ينقل المستخدم إلى قسم رئيسي (callback قبل إعادة التشغيل)."""
    st.session_state["main_tabs"] = label


def _home_service_card(
    col, *, icon, title_ar, title_en, desc, meta,
    button_label, target, button_key, card_key,
):
    with col, st.container(border=True, key=f"card_home_{card_key}"):
        st.markdown(
            f'<div class="home-icon">{icon}</div>'
            f'<div class="home-title">{title_ar}<em>{title_en}</em></div>'
            f'<div class="home-desc">{desc}</div>'
            f'<span class="home-meta">{meta}</span>',
            unsafe_allow_html=True,
        )
        st.button(
            button_label,
            key=button_key,
            on_click=_go_main,
            args=(target,),
            width="stretch",
        )


def render_home_tab(source, market):
    group_label("لوحة البداية", "Start Here", "🏠")
    st.caption(
        "كل الخدمات في مكان واحد — اختر القسم الذي تريد البدء به. | "
        "Everything in one place — pick where to start."
    )

    cards = st.columns(3)
    _home_service_card(
        cards[0],
        icon="📊",
        title_ar="التحليلات الذكية",
        title_en="Smart Analytics",
        desc="كشف الانحرافات، تحليلات الأعمال، التحليل التنبؤي، ومراقبة تغيّر البيانات (Drift).",
        meta="4 أدوات · انحرافات · أعمال · تنبؤي · Drift",
        button_label="افتح التحليلات ←",
        target=NAV_ANALYTICS,
        button_key="home_go_analytics",
        card_key="analytics",
    )
    _home_service_card(
        cards[1],
        icon="🚀",
        title_ar="مخطط الحملات",
        title_en="Campaign Planner",
        desc="قدّر حجم الجمهور والميزانية والنتائج المتوقعة لإعلاناتك قبل الإطلاق.",
        meta="جمهور · ميزانية · نقرات · CPA",
        button_label="افتح الحملات ←",
        target=NAV_CAMPAIGN,
        button_key="home_go_campaign",
        card_key="campaign",
    )
    _home_service_card(
        cards[2],
        icon="💬",
        title_ar="اسأل بياناتك",
        title_en="Ask Your Data",
        desc="محادثة عربية/إنجليزية تجيبك بالأرقام من بياناتك، مع بديل محلي.",
        meta="بديل محلي · بدون مفتاح",
        button_label="افتح المحادثة ←",
        target=NAV_CHAT,
        button_key="home_go_chat",
        card_key="chat",
    )

    st.write("")
    with st.container(border=True, key="card_home_data"):
        st.markdown(
            '<div class="section-title">📂 بياناتك الحالية | Your Data</div>',
            unsafe_allow_html=True,
        )
        try:
            df, _meta = _read_cached(source["name"], source["payload"])
            quality = _quality_cached(source["name"], source["payload"])
            stats = st.columns(4)
            kpi_card(stats[0], f"{len(df):,}", "الصفوف", "Rows", 0.0)
            kpi_card(stats[1], df.shape[1], "الأعمدة", "Columns", 0.05)
            kpi_card(stats[2], f"{quality['score']}/100", "جودة البيانات", "Quality", 0.10)
            kpi_card(stats[3], market.get("currency", "JOD"), "العملة", "Currency", 0.15)
            if source.get("origin") == "demo":
                st.caption(
                    "📊 وضع تجريبي — ارفع ملفًا أو اربط قاعدة بيانات من الشريط الجانبي "
                    "لتطبيق التحليلات على بياناتك. | Demo mode — upload or connect a database."
                )
            else:
                st.caption(f"✅ المصدر النشط: {source['label']} · يمكنك تغييره من الشريط الجانبي.")
        except Exception as err:
            st.caption(f"⚠️ تعذّر قراءة مصدر البيانات | Could not read source: {err}")
