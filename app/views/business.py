"""Business analytics tab and the AI financial report card."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import ai_report as ai
import pdf_export as pdf
from charts import (
    plot_category_donut,
    plot_daily_anomalies,
    plot_expense_vs_revenue,
    plot_forecast,
    plot_pareto,
    plot_revenue_profit,
    plot_status,
    plot_weekday,
)
from services import _analyze_cached
from ui import _money, group_label, kpi_card, render_quality_card

ROLE_LABELS = {
    "date": "التاريخ | Date",
    "revenue": "الإيراد | Revenue",
    "expense": "المصاريف | Expenses",
    "profit": "الربح | Profit",
    "quantity": "الكمية | Quantity",
    "client": "العميل | Client",
    "category": "الخدمة/الفئة | Category",
    "status": "الحالة | Status",
}


def render_ai_report(result, market, ai_conf):
    """بطاقة التقرير المالي الذكي (ذكاء اصطناعي مع بديل محلي)."""
    with st.container(border=True, key="card_ai"):
        st.markdown(
            '<div class="section-title">🤖 التقرير المالي الذكي | AI Financial Report</div>',
            unsafe_allow_html=True,
        )

        context = ai.build_context(result, market)
        signature = ai.signature(context)
        api_key = ai.resolve_api_key(ai_conf.get("key"))

        col_btn, col_info = st.columns([1.15, 2])
        with col_btn:
            generate = st.button(
                "✨ توليد التقرير | Generate",
                width="stretch",
                key="ai_generate",
            )
        with col_info:
            if api_key:
                st.caption("✅ الذكاء الاصطناعي جاهز.")
            else:
                st.caption(
                    "⚪ وضع التحليل المحلي — سيُبنى التقرير من أرقام بياناتك تلقائيًا."
                )

        state = st.session_state
        report_box = st.empty()
        auto_run = bool(ai_conf.get("auto")) and state.get("ai_sig") != signature

        if generate or auto_run:
            # معاينة محلية فورية أثناء انتظار التحسين الذكي (بدل شاشة معلّقة)
            with report_box.container():
                st.caption("⏳ معاينة محلية فورية — جارٍ تحسينها بالذكاء الاصطناعي...")
                st.markdown(ai.fallback_report(context))

            try:
                report, meta = ai.generate_report(context, api_key, ai_conf["model"])
            except Exception as err:
                report = ai.fallback_report(context)
                meta = {"source": "local", "model": None,
                        "note": f"{type(err).__name__}: {err}"}

            state["ai_report"] = report
            state["ai_meta"] = meta
            state["ai_sig"] = signature

        report = state.get("ai_report")
        if report and state.get("ai_sig") == signature:
            meta = state.get("ai_meta") or {}
            with report_box.container():
                if meta.get("source") in ("ai", "gemini"):
                    st.caption("✅ تم التوليد بالذكاء الاصطناعي")
                else:
                    st.caption(f"ℹ️ تقرير محلي احتياطي — {meta.get('note', '')}")
                st.markdown(report)
            col_md, col_pdf = st.columns(2)
            with col_md:
                st.download_button(
                    "⬇️ تحميل Markdown | Download Markdown",
                    data=report.encode("utf-8"),
                    file_name="ai_financial_report.md",
                    mime="text/markdown",
                    width="stretch",
                    key="ai_dl_md",
                )
            with col_pdf:
                try:
                    pdf_bytes = pdf.ai_report_pdf(context, report, meta)
                    st.download_button(
                        "📄 تحميل PDF | Download PDF",
                        data=pdf_bytes,
                        file_name="ai_financial_report.pdf",
                        mime="application/pdf",
                        width="stretch",
                        key="ai_dl_pdf",
                    )
                except Exception as err:
                    st.caption(
                        f"⚠️ تعذّر إنشاء PDF | PDF failed: "
                        f"{type(err).__name__}: {err}"
                    )
        else:
            with report_box.container():
                st.info(
                    "اضغط **توليد التقرير** ليحلّل الذكاء الاصطناعي بياناتك ويكتب ملخصًا تنفيذيًا "
                    "ومؤشرات ومخاطر وتوقّعًا وتوصيات. | "
                    "Click Generate to create the AI financial report."
                )


def render_business_tab(market, ai_conf, source):
    group_label("تحليلات المبيعات والمصاريف والربح", "Sales, Expenses & Profit", "📊")

    name, payload, source_label = source["name"], source["payload"], source["label"]

    # ---------------- 1) قراءة الملف وتحليله ----------------
    try:
        df, _meta, result = _analyze_cached(
            name, payload,
            market["vat_rate"], market["currency"],
            market["target_margin"], market["forecast_days"],
        )
    except Exception as err:
        st.error(f"⚠️ تعذر قراءة/تحليل الملف | Could not read or analyze: `{type(err).__name__}: {err}`")
        return

    if df.empty:
        st.warning("الملف فارغ | The file is empty.")
        return

    st.session_state["biz_shared_df"] = df
    st.session_state["biz_shared_source"] = source_label

    # ---------------- 1) الأعمدة المكتشفة وجودة البيانات ----------------
    with st.container(border=True, key="card_source"):
        st.markdown(
            f'<span class="badge-orange">{source_label} — {len(df):,} صف / rows</span>',
            unsafe_allow_html=True,
        )

        roles = result["roles"]
        chips = "".join(
            f'<span class="col-chip"><b>{ROLE_LABELS[key]}</b>{roles.get(key) or "—"}</span>'
            for key in ROLE_LABELS
        )
        st.markdown(f'<div class="col-chips">{chips}</div>', unsafe_allow_html=True)

        if not result["has_expense"]:
            st.info(
                "ℹ️ لم يُعثر على عمود تكلفة/مصاريف، لذلك الربح = الإيراد. "
                "أضف عمودًا مثل `Cost` أو `Expense` لتحليل الربح الحقيقي. | "
                "No expense column found; profit equals revenue."
            )
        if not result["has_date"]:
            st.info(
                "ℹ️ لم يُعثر على عمود تاريخ، لذلك الرسوم الزمنية والتوقّع غير متاحة. | "
                "No date column found; time series and forecasting are unavailable."
            )

        render_quality_card(name, payload)

    kpis = result["kpis"]
    cur = market["currency"]
    daily = result["daily"]
    monthly = result["monthly"]
    score = result["score"]

    # ---------------- 2) المؤشرات الرئيسية ----------------
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, _money(kpis["revenue"], cur), "إجمالي الإيراد", "Total Revenue", 0.0)
    kpi_card(c2, _money(kpis["expense"], cur), "إجمالي المصاريف", "Total Expenses", 0.05)
    kpi_card(c3, _money(kpis["profit"], cur), "صافي الربح", "Net Profit", 0.1)
    kpi_card(c4, f"{kpis['margin']*100:.1f}%", "هامش الربح", "Profit Margin", 0.15)

    c5, c6, c7, c8 = st.columns(4)
    growth = kpis["growth"]
    kpi_card(c5, f"{growth*100:+.1f}%" if growth is not None else "—",
             "نمو الإيراد", "Revenue Growth", 0.2)
    kpi_card(c6, _money(kpis["avg_ticket"], cur), "متوسط الفاتورة", "Average Ticket", 0.25)
    kpi_card(c7, f"{kpis['collection_rate']*100:.0f}%", "نسبة التحصيل", "Collection Rate", 0.3)
    kpi_card(c8, f"{kpis['orders']:,}", "عدد الفواتير", "Invoices", 0.35)

    st.write("")

    # ---------------- 3) التقرير الذكي ----------------
    render_ai_report(result, market, ai_conf)

    # ---------------- 4) الصحة المالية ----------------
    s1, s2 = st.columns(2)
    with s1, st.container(border=True, key="card_score"):
        st.markdown(
            '<div class="section-title">🎯 تقييم صحة الأعمال | Business Health Score</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="score-big">{score["score"]}<span> / 100</span></div>'
            f'<div class="score-label">{score["label"]}</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, score["score"] / 100.0))
        st.caption(
            f"هامش الربح {score['margin_score']}/35 · "
            f"المصاريف {score['expense_score']}/25 · "
            f"التحصيل {score['collection_score']}/25 · "
            f"التركّز {score['risk_score']}/15"
        )
    with s2:  # noqa: SIM117 — عمود ثم بطاقة: بنية Streamlit متعمدة
        with st.container(border=True, key="card_breakeven"):
            st.markdown(
                '<div class="section-title">⚖️ نقطة التعادل والضريبة | Break-even & VAT</div>',
                unsafe_allow_html=True,
            )
            be = result["break_even"]
            rows = [
                ("الإيراد اليومي المطلوب | Daily break-even", _money(be["daily"], cur)),
                ("الإيراد الشهري المطلوب | Monthly break-even", _money(be["monthly"], cur)),
                (f"ضريبة المبيعات {market['vat_rate']*100:.0f}% | VAT amount", _money(kpis["vat_amount"], cur)),
                ("الإيراد بدون ضريبة | Revenue excl. VAT", _money(kpis["revenue_ex_vat"], cur)),
            ]
            for label, value in rows:
                st.markdown(
                    f'<div class="be-row"><span>{label}</span><strong>{value}</strong></div>',
                    unsafe_allow_html=True,
                )

    # ---------------- 5) الأداء الزمني ----------------
    if not daily.empty:
        group_label("الأداء الزمني", "Time Performance", "📈")
        with st.container(border=True, key="card_rev_profit"):
            st.plotly_chart(plot_revenue_profit(daily), width="stretch")

        if not monthly.empty:
            with st.container(border=True, key="card_exp_rev"):
                st.plotly_chart(plot_expense_vs_revenue(monthly), width="stretch")

        f1, f2 = st.columns(2)
        with f1:
            if not result["forecast"].empty:
                with st.container(border=True, key="card_forecast"):
                    st.plotly_chart(plot_forecast(daily, result["forecast"]), width="stretch")
        with f2, st.container(border=True, key="card_daily_anom"):
            st.plotly_chart(plot_daily_anomalies(daily), width="stretch")

    # ---------------- 6) تركيبة الأعمال ----------------
    group_label("تركيبة الأعمال", "Business Mix", "🧩")
    d1, d2 = st.columns(2)
    with d1:
        if not result["services"].empty:
            with st.container(border=True, key="card_services"):
                st.plotly_chart(plot_category_donut(result["services"]), width="stretch")
    with d2:
        if not result["status"].empty:
            with st.container(border=True, key="card_status"):
                st.plotly_chart(plot_status(result["status"]), width="stretch")

    if not result["clients"].empty:
        with st.container(border=True, key="card_pareto"):
            st.plotly_chart(plot_pareto(result["clients"]), width="stretch")

    if not result["weekday"].empty:
        with st.container(border=True, key="card_weekday"):
            st.plotly_chart(plot_weekday(result["weekday"]), width="stretch")

    # ---------------- 7) الجداول ----------------
    if not monthly.empty or not daily.empty:
        group_label("الجداول التفصيلية", "Detailed Tables", "📋")

    if not monthly.empty:
        with st.container(border=True, key="card_monthly"):
            st.markdown(
                '<div class="section-title">🗓️ الملخص الشهري | Monthly Summary</div>',
                unsafe_allow_html=True,
            )
            table = monthly[["label", "revenue", "expense", "profit", "margin", "orders"]].copy()
            table.columns = ["الشهر | Month", "الإيراد | Revenue", "المصاريف | Expenses",
                             "الربح | Profit", "الهامش | Margin", "الفواتير | Invoices"]
            table["الهامش | Margin"] = (table["الهامش | Margin"] * 100).round(1)
            st.dataframe(table, width="stretch", hide_index=True)

    if not daily.empty:
        anomalous = daily[daily["is_anomaly"]].copy()
        with st.container(border=True, key="card_anom_days"):
            st.markdown(
                '<div class="section-title">🔎 الأيام الشاذة في الإيراد | Anomalous Revenue Days</div>',
                unsafe_allow_html=True,
            )
            if anomalous.empty:
                st.caption("لا توجد أيام شاذة بهذا الملف. | No anomalous days detected.")
            else:
                view = anomalous[["date", "revenue", "expense", "profit", "z_score"]].copy()
                view.columns = ["التاريخ | Date", "الإيراد | Revenue", "المصاريف | Expenses",
                                "الربح | Profit", "Z-Score"]
                view["التاريخ | Date"] = pd.to_datetime(view["التاريخ | Date"]).dt.strftime("%Y-%m-%d")
                view["Z-Score"] = view["Z-Score"].round(2)
                st.dataframe(view, width="stretch", hide_index=True)

    # ---------------- 8) التصدير ----------------
    export = result["prepared"].copy()
    export["date"] = export["_date"]
    export = export.drop(columns=["_date"])
    export.columns = ["revenue", "expense", "profit", "quantity", "status", "client", "category", "date"]

    with st.container(border=True, key="card_export"):
        st.markdown(
            '<div class="section-title">⬇️ تصدير النتائج | Export Results</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "حمّل البيانات المعالجة (CSV) أو تقرير التحليل الكامل (PDF) بكل المؤشرات والجداول. | "
            "Download the processed data (CSV) or the full analysis report (PDF)."
        )

        col_csv, col_pdf = st.columns(2)
        with col_csv:
            st.download_button(
                "🗂️ تحميل CSV | Download CSV",
                data=export.to_csv(index=False).encode("utf-8-sig"),
                file_name="business_analytics_results.csv",
                mime="text/csv",
                width="stretch",
                key="biz_dl_csv",
            )
        with col_pdf:
            try:
                analysis_bytes = pdf.analysis_pdf(result, market, source_label)
                st.download_button(
                    "📄 تحميل التحليل الكامل PDF | Download Full PDF",
                    data=analysis_bytes,
                    file_name="business_full_analysis.pdf",
                    mime="application/pdf",
                    width="stretch",
                    key="biz_dl_pdf",
                )
            except Exception as err:
                st.caption(
                    f"⚠️ تعذّر إنشاء PDF | PDF failed: {type(err).__name__}: {err}"
                )
