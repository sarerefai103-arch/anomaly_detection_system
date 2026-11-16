"""Reusable UI pieces: theme loader, KPI cards, section labels, quality card."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import business_analytics as ba
from app_config import CSS_PATH
from services import _quality_cached, _read_cached


def render_quality_card(name: str, payload: bytes):
    """بطاقة جودة البيانات: مؤشر + ملاحظات قابلة للتنفيذ."""
    try:
        quality = _quality_cached(name, payload)
    except Exception as err:
        st.caption(f"⚠️ تعذّر فحص جودة البيانات | Quality check failed: {err}")
        return

    level_icon = {"good": "🟢", "ok": "🟡", "warn": "🟠", "bad": "🔴"}.get(
        quality["level"], "⚪"
    )
    with st.expander(
        f"{level_icon} جودة البيانات | Data Quality — "
        f"{quality['score']}/100 ({quality['label']})"
    ):
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("المؤشر | Score", f"{quality['score']}/100")
        q2.metric("قيم ناقصة | Missing", f"{quality['missing_pct']:.1f}%")
        q3.metric("صفوف مكررة | Duplicates", f"{quality['duplicates']:,}")
        q4.metric("صفوف × أعمدة | Shape", f"{quality['rows']:,} × {quality['columns']}")
        st.progress(min(1.0, quality["score"] / 100.0))

        if quality["issues"]:
            icons = {"bad": "🔴", "warn": "🟠", "info": "🔵"}
            issue_df = pd.DataFrame([
                {
                    "": icons.get(item["severity"], "•"),
                    "الملاحظة | Issue": f"{item['issue_ar']} | {item['issue_en']}",
                    "التفصيل | Detail": item["detail"],
                }
                for item in quality["issues"]
            ])
            st.dataframe(issue_df, width="stretch", hide_index=True)
        else:
            st.success("لا توجد مشكلات جوهرية في جودة البيانات. | No major data quality issues.")

        if quality["worst_missing"]:
            st.caption(
                "أكثر الأعمدة نواقصًا | Most missing: "
                + "، ".join(f"{col} ({count:,})" for col, count in list(quality["worst_missing"].items())[:5])
            )


def load_css():
    with open(CSS_PATH, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def kpi_card(col, value, label_ar, label_en, delay=0.0):
    value = str(value)
    # القيم النصية الطويلة (مثل اسم الخوارزمية) تحتاج خطًا أصغر
    size_cls = " kpi-value-sm" if len(value) > 8 else ""
    col.markdown(
        f"""
        <div class="kpi-card fade-block" style="animation-delay:{delay}s">
            <div class="shine"></div>
            <div class="kpi-value{size_cls}">{value}</div>
            <div class="kpi-label">{label_en}<br>{label_ar}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _money(value, currency):
    symbol = ba.CURRENCIES.get(currency, {}).get("symbol", "")
    return f"{value:,.0f} {symbol}".strip()


def group_label(text_ar, text_en, icon):
    st.markdown(
        f'<div class="group-label"><span>{icon}</span>{text_ar} '
        f'<em>| {text_en}</em></div>',
        unsafe_allow_html=True,
    )


def render_data_strip(source):
    """شريط سياق واحد أعلى التبويبات يوضح المصدر وسلامة البيانات."""
    if source.get("origin") == "demo":
        st.markdown(
            '<div class="source-strip">'
            '<span class="badge-orange">📊 وضع تجريبي | Demo mode</span>'
            '<span class="source-note">ارفع ملفًا أو اربط قاعدة بيانات من الشريط الجانبي ليُطبَّق على جميع التحليلات.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        df, _meta = _read_cached(source["name"], source["payload"])
        quality = _quality_cached(source["name"], source["payload"])
        quality_cls = {
            "good": "badge-green", "ok": "badge-green",
            "warn": "badge-orange", "bad": "badge-red",
        }.get(quality["level"], "badge-orange")
        origin_chip = (
            '<span class="source-meta">🗄️ قاعدة بيانات | Database</span>'
            if source.get("origin") == "database" else ""
        )
        st.markdown(
            '<div class="source-strip">'
            f'<span class="badge-orange">{source["label"]}</span>'
            f'<span class="{quality_cls}">جودة البيانات {quality["score"]}/100 | Quality</span>'
            f'<span class="source-meta">{len(df):,} صف × {df.shape[1]} عمود</span>'
            f'{origin_chip}'
            '<span class="source-meta">يُطبَّق على جميع التحليلات | All analyses</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    except Exception as err:
        st.caption(f"⚠️ تعذّر فحص مصدر البيانات | Source check failed: {err}")
