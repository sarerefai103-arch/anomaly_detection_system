"""Predictive analytics tab."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import ai_report as ai
import forecasting as fc
import pdf_export as pdf
import predictive_analytics as pa
from charts import (
    plot_backtest,
    plot_customer_segments,
    plot_demand_forecast,
    plot_predictive_forecast,
)
from services import _analyze_cached
from ui import group_label


def _pred_kpi(label: str, value: str, hint: str = "", color: str = "#FF7A1A") -> str:
    # القيم الطويلة (مبالغ بعملة أو تسميات ثنائية اللغة) تحتاج حجم خط أصغر
    size_class = " kpi-value-sm" if len(str(value)) > 8 else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value{size_class}" style="color:{color}">{value}</div>'
        + (f'<div class="kpi-hint">{hint}</div>' if hint else "")
        + "</div>"
    )


def render_predictive_tab(market, ai_conf, source):
    group_label("التحليل التنبؤي المدعوم بالذكاء الاصطناعي", "AI-Powered Predictive Analytics", "🔮")

    name, payload, source_label = source["name"], source["payload"], source["label"]

    # ---------------- قراءة الملف وتحليله ----------------
    try:
        df, meta, result = _analyze_cached(
            name, payload,
            market["vat_rate"], market["currency"],
            market["target_margin"], market["forecast_days"],
        )
        pred = pa.run_predictive(result, market)
    except Exception as err:
        st.error(f"⚠️ تعذر التحليل التنبؤي | Predictive analysis failed: `{type(err).__name__}: {err}`")
        return

    st.session_state["biz_shared_df"] = df
    st.session_state["biz_shared_source"] = source_label

    st.caption(
        f"{source_label} — {len(df):,} صف / rows — أفق التوقّع {pred['periods']} يوم. "
        "غيّر الملف من الشريط الجانبي."
    )

    cur = pred["currency"]
    sales = pred["sales"]
    decline = pred["profit_decline"]
    demand = pred["demand"]
    cust = pred["customers"]

    # ---------------- 1) مؤشرات تنبؤية ----------------
    group_label("المؤشرات التنبؤية", "Predictive KPIs", "📈")
    rev = sales["revenue"]
    prof = sales["profit"]
    risk_colors = {"high": "#FF4D4D", "medium": "#FFB066", "low": "#00C853", "healthy": "#00C853"}
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(_pred_kpi(
            "إيراد متوقع | Forecast Revenue",
            pa._money(rev.get("total"), cur),
            f"نمو {pa._pct(rev.get('growth_pct'))} vs الفترة السابقة",
        ), unsafe_allow_html=True)
    with r2:
        st.markdown(_pred_kpi(
            "ربح متوقع | Forecast Profit",
            pa._money(prof.get("total"), cur),
            f"متوسط يومي {pa._money(prof.get('avg_daily'), cur)}",
            "#FFB066",
        ), unsafe_allow_html=True)
    with r3:
        decline_note = (decline.get("note") or "").strip()
        if len(decline_note) > 88:
            decline_note = decline_note[:86].rstrip() + "…"
        st.markdown(_pred_kpi(
            "خطر انخفاض الربح | Profit Decline Risk",
            decline.get("risk", "—"),
            decline_note,
            risk_colors.get(decline.get("risk_level", "unknown"), "#FF7A1A"),
        ), unsafe_allow_html=True)
    with r4:
        at_risk = cust.get("summary", {}).get("at_risk_customers") if cust.get("available") else None
        at_pct = cust.get("summary", {}).get("at_risk_pct") if cust.get("available") else None
        st.markdown(_pred_kpi(
            "عملاء معرّضون للفقدان | At-Risk Customers",
            "—" if at_risk is None else f"{int(at_risk):,}",
            "بيانات العملاء غير متاحة" if at_risk is None else f"{pa._pct(at_pct)} من إجمالي العملاء",
            "#FF4D4D" if (at_risk or 0) else "#888",
        ), unsafe_allow_html=True)

    st.info(
        "ℹ️ التوقعات تقديرية ومبنية على الأنماط التاريخية (اتجاه + موسمية أسبوعية) وقد تتأثر بعوامل خارجية. "
        "| Forecasts are estimates based on historical patterns and may be affected by external factors."
    )

    # ---------------- 2) توقع المبيعات والإيرادات ----------------
    group_label("توقع المبيعات والإيرادات", "Sales & Revenue Forecast", "💰")
    with st.container(border=True, key="card_pred_rev"):
        st.plotly_chart(plot_predictive_forecast(result["daily"], sales, "revenue"), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        if "profit" in result["daily"].columns:
            with st.container(border=True, key="card_pred_profit"):
                st.plotly_chart(plot_predictive_forecast(result["daily"], sales, "profit"), width="stretch")
    with c2:
        if "expense" in result["daily"].columns:
            with st.container(border=True, key="card_pred_expense"):
                st.plotly_chart(plot_predictive_forecast(result["daily"], sales, "expense"), width="stretch")

    # ---------------- 3) تقييم دقة التوقعات (Backtesting) ----------------
    group_label("تقييم دقة التوقعات", "Forecast Accuracy (Backtesting)", "🎯")
    metric_labels = {"revenue": "الإيراد | Revenue", "profit": "الربح | Profit", "expense": "المصاريف | Expense"}
    metric_options = [c for c in ("revenue", "profit", "expense") if c in result["daily"].columns]
    with st.container(border=True, key="card_pred_backtest"):
        if not metric_options or result["daily"].empty:
            st.info("لا توجد بيانات تاريخ كافية لتقييم الدقة. | Not enough dated data for backtesting.")
        else:
            available_days = max(1, len(result["daily"]))
            # يلزم 21 يومًا للتدريب + 3 أيام اختبار على الأقل.
            # min_value يجب أن يكون أصغر من max_value وإلا يرفع Streamlit استثناءً.
            max_test = min(60, available_days - 21)
            if max_test < 4:
                st.info(
                    f"يلزم {21 + 3} يومًا على الأقل لتقييم الدقة؛ "
                    f"المتوفر {available_days} يومًا فقط. "
                    "| At least 25 dated days are required for backtesting."
                )
                bt = {"available": False, "note": ""}
            else:
                bt1, bt2, bt3 = st.columns([1.2, 1.4, 1.4])
                with bt1:
                    bt_metric = st.selectbox(
                        "المؤشر | Metric", metric_options,
                        format_func=lambda m: metric_labels.get(m, m), key="bt_metric",
                    )
                with bt2:
                    # مفتاح ديناميكي حتى لا تحمل قيمة قديمة خارج النطاق الجديد
                    bt_days = st.slider(
                        "أيام الاختبار | Test days", 3, max_test,
                        min(14, max_test), key=f"bt_days_{max_test}",
                    )
                with bt3:
                    st.caption(
                        "يُدرَّب النموذج على كل الأيام ما عدا آخر N يوم، ثم تُقاس دقته عليها. "
                        "MAPE أقل = توقّع أدق."
                    )

                bt = fc.backtest(result["daily"], column=bt_metric, test_days=bt_days)
            if not bt["available"]:
                if bt.get("note"):
                    st.info(bt["note"])
            else:
                best_row = bt["metrics"].iloc[0]
                level_class = {
                    "good": "badge-green", "ok": "badge-orange",
                    "warn": "badge-orange", "bad": "badge-red",
                }.get(bt["reliability_level"], "badge-orange")
                st.markdown(
                    f'<span class="{level_class}">أفضل نموذج | Best model: '
                    f'{bt["best_label"]} — {bt["reliability"]}</span>',
                    unsafe_allow_html=True,
                )
                b1, b2, b3, b4 = st.columns(4)
                mape = best_row.get("mape")
                b1.metric("MAPE", "—" if pd.isna(mape) else f"{float(mape):.1f}%")
                b2.metric("MAE", "—" if pd.isna(best_row.get("mae")) else f"{float(best_row['mae']):,.2f}")
                b3.metric("RMSE", "—" if pd.isna(best_row.get("rmse")) else f"{float(best_row['rmse']):,.2f}")
                bias = best_row.get("bias")
                b4.metric("Bias", "—" if pd.isna(bias) else f"{float(bias):+,.2f}")

                chart_col, table_col = st.columns([1.6, 1])
                with chart_col:
                    st.plotly_chart(plot_backtest(bt["chart"]), width="stretch")
                with table_col:
                    st.markdown("**مقارنة النماذج | Models Comparison**")
                    comp = bt["metrics"][["label", "mae", "rmse", "mape", "bias"]].copy()
                    comp.columns = ["النموذج | Model", "MAE", "RMSE", "MAPE %", "Bias"]
                    st.dataframe(
                        comp.style.format({
                            "MAE": "{:,.2f}", "RMSE": "{:,.2f}",
                            "MAPE %": "{:.1f}", "Bias": "{:+,.2f}",
                        }, na_rep="—"),
                        width="stretch", hide_index=True,
                    )
                st.caption(bt["note"])
                if bt["reliability_level"] in {"warn", "bad"}:
                    st.caption(
                        "ℹ️ الدقة المنخفضة شائعة في الإيراد اليومي المتذبذب؛ "
                        "جرّب بيانات أطول أو مؤشرًا أكثر استقرارًا. | "
                        "Low daily accuracy is common with noisy revenue; try more history."
                    )

    # ---------------- 4) محاكي السيناريوهات (What-if) ----------------
    group_label("محاكي السيناريوهات", "What-if Scenario Simulator", "🧪")
    with st.container(border=True, key="card_pred_whatif"):
        st.caption(
            "غيّر السعر أو الكمية أو المصاريف وشاهد أثرها المتوقع على الإيراد والربح. "
            "| Change price, volume or expenses and see the projected impact."
        )
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            price_change = st.slider("تغيير السعر % | Price change %", -30, 30, 0, 1, key="wi_price")
        with w2:
            volume_change = st.slider("تغيير الكمية % | Volume change %", -30, 50, 0, 1, key="wi_volume")
        with w3:
            expense_change = st.slider("تغيير المصاريف % | Expense change %", -30, 30, 0, 1, key="wi_expense")
        with w4:
            horizon = st.slider("الأفق (يوم) | Horizon (days)", 7, 90, int(market.get("forecast_days", 30)), 1, key="wi_horizon")

        wi = fc.what_if(
            result["daily"], horizon_days=horizon, window_days=30,
            price_change_pct=price_change / 100,
            volume_change_pct=volume_change / 100,
            expense_change_pct=expense_change / 100,
            currency=cur,
        )
        if not wi["available"]:
            st.info(wi.get("note", "بيانات غير كافية للمحاكاة."))
        else:
            base = wi["baseline"]
            scen = wi["scenario"]
            delta = wi["delta"]
            verdict_cls = {"good": "badge-green", "bad": "badge-red"}.get(wi["level"], "badge-orange")
            st.markdown(
                f'<span class="{verdict_cls}">النتيجة | Verdict: {wi["verdict"]}</span>',
                unsafe_allow_html=True,
            )
            s1, s2, s3 = st.columns(3)
            s1.metric(
                "الإيراد المتوقع | Projected Revenue",
                pa._money(scen["revenue"], cur),
                f"{pa._money(delta['revenue'], cur)}",
            )
            s2.metric(
                "الربح المتوقع | Projected Profit",
                pa._money(scen["profit"], cur),
                f"{pa._money(delta['profit'], cur)}",
            )
            s3.metric(
                "هامش الربح | Profit Margin",
                pa._pct(scen["margin_pct"]),
                f"{pa._pct(base['margin_pct'])} أساس | baseline",
            )
            st.caption(wi.get("note", ""))

    # ---------------- 5) توقع الطلب على المنتجات ----------------
    group_label("توقع الطلب على المنتجات/الخدمات", "Product & Service Demand Forecast", "📦")
    if demand.empty:
        st.info("لا تتوفر بيانات منتجات/خدمات كافية لتوقع الطلب. | Insufficient product/service data for demand forecasting.")
    else:
        with st.container(border=True, key="card_pred_demand"):
            st.plotly_chart(plot_demand_forecast(demand), width="stretch")
        with st.container(border=True, key="card_pred_demand_tbl"):
            st.markdown('<div class="section-title">تفاصيل الطلب المتوقع | Demand Details</div>', unsafe_allow_html=True)
            tbl = demand[["trend_icon", "service", "historical_orders", "projected_orders", "projected_revenue", "share_pct", "trend"]].copy()
            tbl.columns = ["", "الخدمة/المنتج | Service", "طلبات تاريخية | Historical", "طلبات متوقعة | Projected",
                           f"إيراد متوقع | Revenue ({cur})", "حصة % | Share", "الاتجاه | Trend"]
            st.dataframe(tbl, width="stretch", hide_index=True)

    # ---------------- 6) توقع انخفاض الأرباح ----------------
    group_label("توقع انخفاض الأرباح", "Profit Decline Prediction", "⚠️")
    with st.container(border=True, key="card_pred_decline"):
        if not decline.get("available"):
            st.info(decline.get("note", "بيانات غير كافية."))
        else:
            level = decline.get("risk_level", "unknown")
            badge_cls = {"high": "badge-red", "medium": "badge-orange", "low": "badge-green", "healthy": "badge-green"}.get(level, "badge-orange")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<span class="{badge_cls}">مستوى الخطر | Risk: {decline.get("risk")}</span>', unsafe_allow_html=True)
                st.metric("ميل الربح اليومي | Daily Profit Slope", f"{decline.get('slope_daily', '—')}")
            with m2:
                st.metric("الربح المتوقع (متوسط يومي) | Projected Avg Profit", pa._money(decline.get("projected_profit"), cur))
                st.metric("الهامش الحديث | Recent Margin", pa._pct(decline.get("recent_margin_pct")))
            with m3:
                dtn = decline.get("days_to_negative")
                st.metric("أيام تقديرية للمنطقة الحرجة | Days to Critical Zone", "—" if dtn is None else f"~{dtn} يوم")
                st.caption(decline.get("note", ""))

            st.markdown("**إشارات الإنذار المبكر | Early Warning Signals**")
            for s in decline.get("signals", []):
                st.markdown(f"- {s}")

    # ---------------- 7) توقع سلوك العملاء ----------------
    group_label("توقع سلوك العملاء", "Customer Behavior Prediction", "👥")
    if not cust.get("available"):
        st.info(cust.get("note", "بيانات العملاء غير متاحة."))
    else:
        sm = cust.get("summary", {})
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("إجمالي العملاء | Customers", f"{int(sm.get('total_customers', 0)):,}")
        k2.metric("معرّضون للفقدان | At Risk", f"{int(sm.get('at_risk_customers', 0)):,} ({pa._pct(sm.get('at_risk_pct'))})")
        k3.metric("حصة أعلى 20% | Top 20% Share", pa._pct(sm.get("top20_value_share_pct")))
        k4.metric("متوسط القيمة | Avg Monetary", pa._money(sm.get("avg_monetary"), cur))

        ch1, ch2 = st.columns([1, 1])
        with ch1:
            fig = plot_customer_segments(cust)
            if fig:
                with st.container(border=True, key="card_pred_seg"):
                    st.plotly_chart(fig, width="stretch")
        with ch2:  # noqa: SIM117 — عمود ثم بطاقة: بنية Streamlit متعمدة
            with st.container(border=True, key="card_pred_cust_sig"):
                st.markdown('<div class="section-title">إشارات سلوكية | Behavior Signals</div>', unsafe_allow_html=True)
                for s in cust.get("signals", []):
                    st.markdown(f"- {s}")
                st.caption(cust.get("note", ""))

        with st.container(border=True, key="card_pred_cust_tbl"):
            st.markdown('<div class="section-title">أهم العملاء (RFM) | Top Customers by RFM</div>', unsafe_allow_html=True)
            seg_df = cust["segments"].head(15).copy()
            seg_df["monetary"] = seg_df["monetary"].round(2)
            view = seg_df[["icon", "segment", "_client", "recency_days", "frequency", "monetary", "rfm_score", "churn_risk"]]
            view.columns = ["", "الشريحة | Segment", "العميل | Customer", "الحداثة (أيام) | Recency", "التكرار | Frequency",
                            f"القيمة | Monetary ({cur})", "RFM", "خطر الفقدان | Churn Risk"]
            st.dataframe(view, width="stretch", hide_index=True)

    # ---------------- 8) رؤى الذكاء الاصطناعي التنبؤية ----------------
    group_label("رؤى الذكاء الاصطناعي التنبؤية", "AI Predictive Insights", "🤖")
    with st.container(border=True, key="card_pred_ai"):
        st.markdown(
            '<div class="section-title">🤖 تقرير تنبؤي ذكي | AI Predictive Report</div>',
            unsafe_allow_html=True,
        )
        ctx = pa.build_predictive_context(pred, result, market)
        try:
            import hashlib
            import json as _json
            sig = hashlib.sha256(_json.dumps(ctx, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        except Exception:
            sig = "pred"
        state = st.session_state
        if state.get("pred_sig") != sig:
            state["pred_sig"] = sig
            state["pred_report"] = None
            state["pred_meta"] = None

        api_key = ai.resolve_api_key(ai_conf.get("api_key"))
        auto = ai_conf.get("auto", False)
        model = ai_conf.get("model", pa.DEFAULT_MODEL)
        cgen, cclr = st.columns([1, 1])
        with cgen:
            gen = st.button("🔮 توليد تقرير تنبؤي بالذكاء الاصطناعي | Generate Predictive AI Report",
                            key="pred_ai_gen", type="primary", width="stretch")
        with cclr:
            if st.button("مسح | Clear", key="pred_ai_clear", width="stretch"):
                state["pred_report"] = None
                state["pred_meta"] = None

        box = st.empty()
        should = gen or (auto and state.get("pred_report") is None)
        if should:
            box.markdown(pa.fallback_insights(ctx))
            try:
                report, meta = pa.generate_ai_insights(ctx, api_key, model)
            except Exception as err:
                report = pa.fallback_insights(ctx)
                meta = {"source": "local", "model": None, "note": f"{type(err).__name__}: {err}"}
            state["pred_report"] = report
            state["pred_meta"] = meta

        report = state.get("pred_report")
        if report and state.get("pred_sig") == sig:
            meta = state.get("pred_meta") or {}
            with box.container():
                if meta.get("source") in ("ai", "gemini"):
                    st.caption("✅ تم التوليد بالذكاء الاصطناعي")
                else:
                    st.caption(f"ℹ️ تقرير محلي احتياطي — {meta.get('note', '')}")
                st.markdown(report)
            cmd, cpd = st.columns(2)
            with cmd:
                st.download_button(
                    "⬇️ تحميل Markdown | Download Markdown", data=report.encode("utf-8"),
                    file_name="predictive_ai_report.md", mime="text/markdown",
                    width="stretch", key="pred_dl_md",
                )
            with cpd:
                try:
                    pdf_bytes = pdf.ai_report_pdf(
                        {**ctx, "data": {"first_day": None, "last_day": None},
                         "totals": {"revenue": rev.get("total"), "profit": prof.get("total"),
                                    "margin_pct": pa._num((result["kpis"].get("margin", 0) or 0)*100),
                                    "orders": result["kpis"].get("orders")},
                         "market_health": {"score": result.get("score", {}).get("score", "—")},
                         "vat_rate": market.get("vat_rate", 0)},
                        report, meta,
                    )
                    st.download_button(
                        "📄 تحميل PDF | Download PDF", data=pdf_bytes,
                        file_name="predictive_ai_report.pdf", mime="application/pdf",
                        width="stretch", key="pred_dl_pdf",
                    )
                except Exception as err:
                    st.caption(f"⚠️ تعذّر إنشاء PDF | PDF failed: {type(err).__name__}: {err}")
