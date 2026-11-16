"""Campaign planning tab (works without a loaded dataset)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import campaign_planner as cp
from ui import group_label, kpi_card


def _cp_default_price(platform_key: str, model: str) -> float:
    """السعر المرجعي الافتراضي (USD) حسب المنصة ونموذج التسعير."""
    item = cp.PLATFORMS.get(platform_key) or cp.PLATFORMS["instagram"]
    key = "cpc" if str(model).upper() == "CPC" else "cpm"
    return float(item[key])


def _sync_cp_price():
    """يزامن سعر CPM/CPC مع المنصة والنموذج المختار."""
    st.session_state["cp_price"] = _cp_default_price(
        st.session_state.get("cp_platform", "instagram"),
        st.session_state.get("cp_model", "CPM"),
    )


def render_campaign_tab(market):
    """خدمة تخطيط الحملات: مدخلات، تقدير جمهور، ميزانية، ونتائج متوقعة."""
    group_label("مخطط الحملات الإعلانية", "Campaign Planner", "🚀")

    st.markdown(
        '<div class="source-strip">'
        '<span class="badge-orange">🚀 خدمة مستقلة | Standalone service</span>'
        '<span class="source-note">لا تحتاج رفع بيانات — تعمل مباشرة. | No data upload required.</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "أدخل بيانات الحملة لتحصل على تقدير لحجم الجمهور، والميزانية المطلوبة، "
        "والنتائج المتوقعة (ظهور / نقرات / CPA). | "
        "Enter your campaign details to estimate audience size, budget and expected results."
    )

    if "cp_platform" not in st.session_state:
        st.session_state["cp_platform"] = "instagram"
    if "cp_objective" not in st.session_state:
        st.session_state["cp_objective"] = "engagement"
    if "cp_geo" not in st.session_state:
        st.session_state["cp_geo"] = "jordan"
    if "cp_model" not in st.session_state:
        st.session_state["cp_model"] = "CPM"
    if "cp_price" not in st.session_state:
        st.session_state["cp_price"] = _cp_default_price(
            st.session_state["cp_platform"], st.session_state["cp_model"]
        )
    default_currency = market.get("currency", "JOD")
    if default_currency not in cp.CURRENCIES:
        default_currency = "JOD"
    if "cp_currency" not in st.session_state:
        st.session_state["cp_currency"] = default_currency
    if "cp_budget" not in st.session_state:
        st.session_state["cp_budget"] = 500.0

    # ---------------- 1) مدخلات الحملة ----------------
    with st.container(border=True, key="card_campaign_inputs"):
        st.markdown(
            '<div class="section-title">🎯 مدخلات الحملة | Campaign Inputs</div>',
            unsafe_allow_html=True,
        )
        col_platform, col_objective, col_geo = st.columns(3)
        with col_platform:
            platform_key = st.selectbox(
                "المنصة | Platform",
                list(cp.PLATFORMS.keys()),
                format_func=lambda key: (
                    f"{cp.PLATFORMS[key]['icon']} "
                    f"{cp.PLATFORMS[key]['label_ar']} | {cp.PLATFORMS[key]['label_en']}"
                ),
                key="cp_platform",
                on_change=_sync_cp_price,
            )
        with col_objective:
            objective_key = st.selectbox(
                "الهدف التسويقي | Objective",
                list(cp.OBJECTIVES.keys()),
                format_func=lambda key: (
                    f"{cp.OBJECTIVES[key]['icon']} "
                    f"{cp.OBJECTIVES[key]['label_ar']} | {cp.OBJECTIVES[key]['label_en']}"
                ),
                key="cp_objective",
            )
        with col_geo:
            geo_key = st.selectbox(
                "النطاق الجغرافي | Location",
                list(cp.GEOS.keys()),
                format_func=lambda key: (
                    f"{cp.GEOS[key]['label_ar']} | {cp.GEOS[key]['label_en']}"
                ),
                key="cp_geo",
            )

        col_age, col_interests = st.columns([2, 3])
        with col_age:
            age_min, age_max = st.slider(
                "الفئة العمرية | Age range",
                min_value=13, max_value=65, value=(18, 35), step=1,
                key="cp_age",
            )
        with col_interests:
            interests = st.multiselect(
                "الاهتمامات | Interests",
                cp.INTEREST_OPTIONS,
                default=[],
                key="cp_interests",
                help="كل اهتمام إضافي يضيّق الجمهور ويرفع دقة الاستهداف.",
            )

        keywords_text = st.text_area(
            "الكلمات المفتاحية | Keywords",
            placeholder="مثال: عبايات، توصيل مجاني، إربد",
            height=80,
            key="cp_keywords",
            help="اكتب الكلمات مفصولة بفاصلة أو سطر جديد.",
        )
        keywords = cp.parse_keywords(keywords_text)

        platform_meta = cp.PLATFORMS[platform_key]
        st.caption(
            f"ℹ️ {platform_meta['icon']} {platform_meta['note_ar']} | "
            f"CPM مرجعي ≈ {platform_meta['cpm']:.2f} USD · "
            f"CPC مرجعي ≈ {platform_meta['cpc']:.2f} USD"
        )

    # ---------------- 2) حاسبة الجمهور ----------------
    audience = cp.estimate_audience(
        platform_key, geo_key, age_min, age_max, interests, keywords
    )

    st.write("")
    with st.container(border=True, key="card_campaign_audience"):
        st.markdown(
            '<div class="section-title">👥 حاسبة الجمهور المستهدف | Audience Estimator</div>',
            unsafe_allow_html=True,
        )
        aud_cols = st.columns(4)
        kpi_card(aud_cols[0], f"{audience['potential_reach']:,}", "الوصول المحتمل", "Potential Reach", 0.0)
        kpi_card(aud_cols[1], f"{audience['age_audience']:,.0f}", "بعد تصفية العمر", "Age Audience", 0.05)
        kpi_card(aud_cols[2], f"{audience['platform_audience']:,.0f}", "مستخدمو المنصة", "Platform Audience", 0.10)
        kpi_card(aud_cols[3], f"{audience['interest_factor'] * 100:,.0f}%", "معامل التضييق", "Interest Factor", 0.15)

        with st.expander("📋 تفاصيل التقدير خطوة بخطوة | Estimation details"):
            st.dataframe(
                pd.DataFrame(audience["steps"]),
                width="stretch",
                hide_index=True,
            )

    # ---------------- 3) الميزانية والتسعير ----------------
    st.write("")
    with st.container(border=True, key="card_campaign_budget"):
        st.markdown(
            '<div class="section-title">💰 الميزانية والتسعير | Budget & Pricing</div>',
            unsafe_allow_html=True,
        )
        col_currency, col_budget, col_model, col_price = st.columns(4)
        with col_currency:
            currency = st.selectbox(
                "العملة | Currency",
                list(cp.CURRENCIES.keys()),
                format_func=lambda code: (
                    f"{code} — {cp.CURRENCIES[code]['name_ar']}"
                ),
                key="cp_currency",
            )
        with col_budget:
            budget = st.number_input(
                "الميزانية الإجمالية | Total budget",
                min_value=0.0,
                max_value=10_000_000.0,
                step=50.0,
                format="%.2f",
                key="cp_budget",
            )
        with col_model:
            pricing_model = st.radio(
                "نموذج التسعير | Pricing model",
                ["CPM", "CPC"],
                horizontal=True,
                key="cp_model",
                on_change=_sync_cp_price,
            )
        with col_price:
            price_usd = st.number_input(
                f"{pricing_model} (USD)",
                min_value=0.01,
                max_value=10_000.0,
                step=0.10,
                format="%.2f",
                key="cp_price",
            )

        fx_rate = float(cp.CURRENCIES[currency]["per_usd"])
        st.caption(
            f"💱 1 USD = {fx_rate:,.3f} {currency} · "
            f"الميزانية بالدولار ≈ {budget / fx_rate:,.2f} USD | "
            f"Budget in USD ≈ {budget / fx_rate:,.2f}"
        )

    # ---------------- 4) بناء الخطة وحساب النتائج ----------------
    plan = cp.build_plan(
        platform_key=platform_key,
        objective_key=objective_key,
        geo_key=geo_key,
        age_min=age_min,
        age_max=age_max,
        interests=interests,
        keywords=keywords,
        budget=budget,
        currency=currency,
        pricing_model=pricing_model,
        price_usd=price_usd,
        fx_rate=fx_rate,
    )
    results = plan["results"]
    cpa_display = (
        results["cpa_usd"] * fx_rate if results["cpa_usd"] is not None else None
    )

    st.write("")
    with st.container(border=True, key="card_campaign_results"):
        st.markdown(
            '<div class="section-title">📈 النتائج المتوقعة | Expected Results</div>',
            unsafe_allow_html=True,
        )
        res_cols = st.columns(5)
        kpi_card(res_cols[0], f"{results['impressions']:,.0f}", "الظهور", "Impressions", 0.0)
        kpi_card(res_cols[1], f"{results['clicks']:,.0f}", "النقرات", "Clicks", 0.05)
        kpi_card(res_cols[2], f"{results['conversions']:,.0f}", "التحويلات", "Conversions", 0.10)
        kpi_card(
            res_cols[3],
            cp.format_money(cpa_display, currency),
            "تكلفة العميل المحتمل",
            "CPA",
            0.15,
        )
        freq_text = (
            f"{results['frequency']:,.2f}" if results["frequency"] is not None else "—"
        )
        kpi_card(res_cols[4], freq_text, "متوسط التكرار", "Frequency", 0.20)

        st.caption(
            f"CTR {cp.format_pct(results['ctr'])} · CVR {cp.format_pct(results['cvr'])} · "
            f"CPM {cp.format_money(results['cpm_usd'], 'USD')} · "
            f"CPC {cp.format_money(results['cpc_usd'], 'USD')}"
        )

        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.plotly_chart(cp.funnel_figure(plan), width="stretch")
        with chart_cols[1]:
            st.plotly_chart(cp.allocation_figure(plan), width="stretch")

    # ---------------- 5) السيناريوهات ----------------
    st.write("")
    with st.container(border=True, key="card_campaign_scenarios"):
        st.markdown(
            '<div class="section-title">🔀 السيناريوهات | Scenarios</div>',
            unsafe_allow_html=True,
        )
        scenario_rows = [
            {
                "السيناريو | Scenario": f"{item['label_ar']} | {item['label_en']}",
                "الظهور | Impressions": f"{item['impressions']:,.0f}",
                "النقرات | Clicks": f"{item['clicks']:,.0f}",
                "التحويلات | Conversions": f"{item['conversions']:,.0f}",
                "CPA": cp.format_money(item["cpa_display"], currency),
                "التكرار | Frequency": (
                    f"{item['frequency']:,.2f}" if item["frequency"] is not None else "—"
                ),
            }
            for item in plan["scenarios"]
        ]
        st.dataframe(pd.DataFrame(scenario_rows), width="stretch", hide_index=True)

    # ---------------- 6) التوصيات ومعايير المنصات ----------------
    st.write("")
    with st.container(border=True, key="card_campaign_tips"):
        st.markdown(
            '<div class="section-title">✅ توصيات سريعة | Quick Recommendations</div>',
            unsafe_allow_html=True,
        )
        for tip in cp._recommendations(plan):
            st.markdown(f"- {tip}")

        with st.expander("📊 معايير المنصات المرجعية | Platform Benchmarks"):
            benchmark = pd.DataFrame(
                [
                    {
                        "المنصة | Platform": (
                            f"{meta['icon']} {meta['label_ar']} | {meta['label_en']}"
                        ),
                        "CPM (USD)": f"{meta['cpm']:.2f}",
                        "CPC (USD)": f"{meta['cpc']:.2f}",
                        "CTR": f"{meta['ctr'] * 100:.2f}%",
                        "CVR": f"{meta['cvr'] * 100:.2f}%",
                    }
                    for meta in cp.PLATFORMS.values()
                ]
            )
            st.dataframe(benchmark, width="stretch", hide_index=True)

    # ---------------- 7) التصدير ----------------
    st.write("")
    export_cols = st.columns(2)
    file_stem = f"campaign_plan_{pd.Timestamp.today().strftime('%Y%m%d')}"
    export_cols[0].download_button(
        "⬇️ تحميل الخطة (Markdown)",
        data=cp.plan_markdown(plan).encode("utf-8"),
        file_name=f"{file_stem}.md",
        mime="text/markdown",
        key="cp_download_md",
        width="stretch",
    )
    export_cols[1].download_button(
        "⬇️ تحميل الملخص (CSV)",
        data=cp.plan_csv(plan),
        file_name=f"{file_stem}.csv",
        mime="text/csv",
        key="cp_download_csv",
        width="stretch",
    )

    st.caption(
        "⚠️ كل الأرقام تقديرية مبنية على متوسطات السوق، وليست ضمانًا للنتائج الفعلية. "
        "راقب أداء الحملة وعدّل الميزانية والاستهداف أسبوعيًا. | "
        "All figures are estimates based on market averages, not guaranteed results."
    )
