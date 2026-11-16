"""
campaign_planner.py
-------------------
مخطط الحملات الإعلانية: تقدير الجمهور، الميزانية، والنتائج المتوقعة.

كل الأرقام تقديرية ومبنية على متوسطات السوق، وليست ضمانًا للنتائج.
الوحدات الحسابية بالدولار الأمريكي (CPM / CPC)، وتُحوَّل الميزانية وCPA
إلى العملة التي يختارها المستخدم للعرض فقط.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go

# ============================================================
# المنصات — متوسطات مرجعية تقريبية (USD)
# ============================================================
PLATFORMS: dict[str, dict[str, Any]] = {
    "instagram": {
        "label_ar": "إنستغرام", "label_en": "Instagram", "icon": "📸",
        "cpm": 8.00, "cpc": 0.70, "ctr": 0.0120, "cvr": 0.0250,
        "share": 0.55,
        "note_ar": "مناسب للوعي والتفاعل والمنتجات البصرية.",
        "note_en": "Best for awareness, engagement and visual products.",
    },
    "facebook": {
        "label_ar": "فيسبوك", "label_en": "Facebook", "icon": "👍",
        "cpm": 7.00, "cpc": 0.60, "ctr": 0.0100, "cvr": 0.0220,
        "share": 0.62,
        "note_ar": "تغطية واسعة واستهداف دقيق للفئات الأكبر سنًا.",
        "note_en": "Broad reach with strong targeting for older audiences.",
    },
    "tiktok": {
        "label_ar": "تيك توك", "label_en": "TikTok", "icon": "🎵",
        "cpm": 5.00, "cpc": 0.50, "ctr": 0.0150, "cvr": 0.0150,
        "share": 0.45,
        "note_ar": "تكلفة ظهور منخفضة ومحتوى فيديو قصير.",
        "note_en": "Low CPM with short-form video content.",
    },
    "linkedin": {
        "label_ar": "لينكدإن", "label_en": "LinkedIn", "icon": "💼",
        "cpm": 25.00, "cpc": 4.50, "ctr": 0.0060, "cvr": 0.0300,
        "share": 0.15,
        "note_ar": "B2B والوظائف والقطاع المهني — تكلفة أعلى.",
        "note_en": "B2B, hiring and professional audiences — higher cost.",
    },
    "snapchat": {
        "label_ar": "سناب شات", "label_en": "Snapchat", "icon": "👻",
        "cpm": 6.00, "cpc": 0.55, "ctr": 0.0130, "cvr": 0.0180,
        "share": 0.30,
        "note_ar": "قوي للفئات الشابة في الأردن والخليج.",
        "note_en": "Strong with younger audiences in Jordan and the GCC.",
    },
    "youtube": {
        "label_ar": "يوتيوب", "label_en": "YouTube", "icon": "▶️",
        "cpm": 4.50, "cpc": 0.45, "ctr": 0.0090, "cvr": 0.0200,
        "share": 0.75,
        "note_ar": "وصول واسع عبر الفيديو والـ Shorts.",
        "note_en": "Broad video and Shorts reach.",
    },
    "google": {
        "label_ar": "إعلانات جوجل", "label_en": "Google Ads", "icon": "🔎",
        "cpm": 10.00, "cpc": 1.20, "ctr": 0.0300, "cvr": 0.0350,
        "share": 0.85,
        "note_ar": "نية شراء عالية عبر البحث والـ Display.",
        "note_en": "High purchase intent through Search and Display.",
    },
}

# ============================================================
# الأهداف التسويقية
# ============================================================
OBJECTIVES: dict[str, dict[str, Any]] = {
    "awareness": {
        "label_ar": "الوعي بالعلامة التجارية", "label_en": "Brand Awareness", "icon": "📣",
        "ctr_mult": 0.80, "cvr_mult": 0.50,
    },
    "engagement": {
        "label_ar": "تفاعل", "label_en": "Engagement", "icon": "❤️",
        "ctr_mult": 1.20, "cvr_mult": 0.60,
    },
    "clicks": {
        "label_ar": "نقرات", "label_en": "Clicks", "icon": "👆",
        "ctr_mult": 1.25, "cvr_mult": 0.70,
    },
    "traffic": {
        "label_ar": "زيارات للموقع", "label_en": "Website Traffic", "icon": "🌐",
        "ctr_mult": 1.30, "cvr_mult": 0.80,
    },
    "conversions": {
        "label_ar": "مبيعات / تحويلات", "label_en": "Conversions", "icon": "🛒",
        "ctr_mult": 0.90, "cvr_mult": 1.40,
    },
    "leads": {
        "label_ar": "عملاء محتملون", "label_en": "Leads", "icon": "📝",
        "ctr_mult": 0.80, "cvr_mult": 1.60,
    },
}

# ============================================================
# النطاقات الجغرافية
# ============================================================
GEOS: dict[str, dict[str, Any]] = {
    "jordan": {
        "label_ar": "الأردن", "label_en": "Jordan",
        "population": 11_500_000, "digital_rate": 0.80,
    },
    "irbid": {
        "label_ar": "إربد", "label_en": "Irbid",
        "population": 2_000_000, "digital_rate": 0.78,
    },
    "amman": {
        "label_ar": "عمّان", "label_en": "Amman",
        "population": 4_800_000, "digital_rate": 0.85,
    },
    "zarqa": {
        "label_ar": "الزرقاء", "label_en": "Zarqa",
        "population": 1_500_000, "digital_rate": 0.78,
    },
    "aqaba": {
        "label_ar": "العقبة", "label_en": "Aqaba",
        "population": 250_000, "digital_rate": 0.80,
    },
    "mena": {
        "label_ar": "المنطقة العربية", "label_en": "MENA",
        "population": 500_000_000, "digital_rate": 0.55,
    },
    "gcc": {
        "label_ar": "دول الخليج", "label_en": "GCC",
        "population": 60_000_000, "digital_rate": 0.85,
    },
    "global": {
        "label_ar": "عالمي", "label_en": "Global",
        "population": 8_000_000_000, "digital_rate": 0.67,
    },
}

# فئات عمرية تقريبية وحصة كل فئة من الجمهور الرقمي
AGE_BANDS = [
    (13, 17, 0.07),
    (18, 24, 0.19),
    (25, 34, 0.27),
    (35, 44, 0.20),
    (45, 54, 0.14),
    (55, 64, 0.09),
    (65, 65, 0.04),
]

INTEREST_OPTIONS = [
    "التجارة الإلكترونية | E-commerce",
    "الأزياء | Fashion",
    "الجمال والعناية | Beauty",
    "الطعام والمطاعم | Food & Dining",
    "السفر | Travel",
    "التقنية | Technology",
    "الصحة واللياقة | Health & Fitness",
    "العقارات | Real Estate",
    "التعليم | Education",
    "السيارات | Automotive",
    "الرياضة | Sports",
    "الترفيه | Entertainment",
    "الأعمال | Business",
    "الأسرة والأطفال | Family & Parenting",
]

CURRENCIES: dict[str, dict[str, Any]] = {
    "JOD": {"name_ar": "دينار أردني", "name_en": "Jordanian Dinar", "per_usd": 0.709},
    "USD": {"name_ar": "دولار أمريكي", "name_en": "US Dollar", "per_usd": 1.000},
}

SCENARIOS = [
    ("conservative", "متحفظ", "Conservative", 0.70, 0.70, 1.15),
    ("expected", "متوقع", "Expected", 1.00, 1.00, 1.00),
    ("optimistic", "متفائل", "Optimistic", 1.30, 1.30, 0.85),
]

BUDGET_ALLOCATION = [
    ("prospecting", "جمهور جديد | Prospecting", 0.70),
    ("retargeting", "إعادة استهداف | Retargeting", 0.20),
    ("testing", "اختبار إبداعي | Creative Testing", 0.10),
]

DARK_TEMPLATE = go.layout.Template(
    layout={
        "paper_bgcolor": "#1C1C21",
        "plot_bgcolor": "#1C1C21",
        "font": {"color": "#F2F2F2"},
        "xaxis": {"gridcolor": "#2A2A30"},
        "yaxis": {"gridcolor": "#2A2A30"},
        "colorway": ["#FF7A1A", "#FFB066", "#FF9A4D", "#5A5A64"],
    }
)


# ============================================================
# أدوات مساعدة
# ============================================================
def age_share(age_min: int, age_max: int) -> float:
    """حصة الفئة العمرية من الجمهور الرقمي (0..1)."""
    lo_bound = max(13, min(65, int(age_min)))
    hi_bound = max(lo_bound, min(65, int(age_max)))
    total = 0.0
    for lo, hi, share in AGE_BANDS:
        overlap = max(0, min(hi, hi_bound) - max(lo, lo_bound) + 1)
        total += share * overlap / (hi - lo + 1)
    return max(0.0, min(1.0, total))


def parse_keywords(text: str) -> list[str]:
    """يحوّل نص الكلمات المفتاحية إلى قائمة فريدة."""
    raw = re.split(r"[,،;\n]+", text or "")
    seen: list[str] = []
    for item in raw:
        word = item.strip()
        if word and word.lower() not in {w.lower() for w in seen}:
            seen.append(word)
    return seen


def format_int(value: float | int | None) -> str:
    """رقم بفواصل الآلاف."""
    return "—" if value is None else f"{value:,.0f}"


def format_money(value: float | None, currency: str = "USD") -> str:
    """مبلغ بعشرتين عشريتين + رمز العملة."""
    if value is None:
        return "—"
    return f"{value:,.2f} {currency}"


def format_pct(value: float | None) -> str:
    """نسبة ككسر (0.25) إلى نص مئوي."""
    return "—" if value is None else f"{value * 100:,.2f}%"


# ============================================================
# 1) حاسبة الجمهور
# ============================================================
def estimate_audience(
    platform_key: str,
    geo_key: str,
    age_min: int,
    age_max: int,
    interests: list[str] | None = None,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    """تقدير حجم الجمهور المحتمل بناءً على المنصة والجغرافيا والعمر والاهتمامات."""
    platform = PLATFORMS[platform_key]
    geo = GEOS[geo_key]
    interests = interests or []
    keywords = keywords or []

    digital = float(geo["population"]) * float(geo["digital_rate"])
    platform_audience = digital * float(platform["share"])
    share = age_share(age_min, age_max)
    age_audience = platform_audience * share

    interest_count = len(interests)
    keyword_count = len(keywords)
    interest_factor = max(0.15, 0.85 ** (interest_count + 0.60 * keyword_count))
    potential_reach = max(0.0, age_audience * interest_factor)

    steps = [
        {
            "المرحلة | Stage": "سكان المنطقة | Population",
            "العدد | People": int(geo["population"]),
            "الملاحظة | Note": f"{geo['label_ar']} / {geo['label_en']}",
        },
        {
            "المرحلة | Stage": "الجمهور الرقمي | Digital audience",
            "العدد | People": int(digital),
            "الملاحظة | Note": f"نسبة الاستخدام الرقمي {geo['digital_rate'] * 100:.0f}%",
        },
        {
            "المرحلة | Stage": "مستخدمو المنصة | Platform audience",
            "العدد | People": int(platform_audience),
            "الملاحظة | Note": f"{platform['label_ar']} — حصة تقديرية {platform['share'] * 100:.0f}%",
        },
        {
            "المرحلة | Stage": "الفئة العمرية | Age group",
            "العدد | People": int(age_audience),
            "الملاحظة | Note": f"{age_min}–{age_max} سنة — حصة {share * 100:.1f}%",
        },
        {
            "المرحلة | Stage": "الاهتمامات | Interests",
            "العدد | People": int(potential_reach),
            "الملاحظة | Note": (
                f"{interest_count} اهتمام و{keyword_count} كلمة مفتاحية — "
                f"معامل التضييق {interest_factor * 100:.0f}%"
            ),
        },
    ]

    return {
        "population": float(geo["population"]),
        "digital": digital,
        "digital_rate": float(geo["digital_rate"]),
        "platform_audience": platform_audience,
        "platform_share": float(platform["share"]),
        "age_share": share,
        "age_audience": age_audience,
        "interest_factor": interest_factor,
        "interest_count": interest_count,
        "keyword_count": keyword_count,
        "potential_reach": round(potential_reach),
        "steps": steps,
        "geo_ar": geo["label_ar"],
        "geo_en": geo["label_en"],
    }


# ============================================================
# 2) حاسبة التكلفة والنتائج
# ============================================================
def effective_rates(platform_key: str, objective_key: str) -> tuple[float, float]:
    """CTR وCVR الفعليان بعد تعديل هدف الحملة."""
    platform = PLATFORMS[platform_key]
    objective = OBJECTIVES[objective_key]
    ctr = float(platform["ctr"]) * float(objective["ctr_mult"])
    cvr = float(platform["cvr"]) * float(objective["cvr_mult"])
    return max(0.0, ctr), max(0.0, cvr)


def calculate_results(
    budget_usd: float,
    pricing_model: str,
    price_usd: float,
    ctr: float,
    cvr: float,
    potential_reach: int,
) -> dict[str, Any]:
    """يحسب الظهور والنقرات والتحويلات وCPA من الميزانية ونموذج التسعير."""
    budget_usd = max(0.0, float(budget_usd))
    price = max(0.0001, float(price_usd))
    ctr = max(0.0, min(0.95, float(ctr)))
    cvr = max(0.0, min(0.95, float(cvr)))

    if pricing_model.upper() == "CPC":
        clicks = budget_usd / price
        impressions = clicks / ctr if ctr > 0 else 0.0
        cpm = budget_usd / impressions * 1000 if impressions > 0 else None
        cpc = price
    else:  # CPM
        impressions = budget_usd / price * 1000
        clicks = impressions * ctr
        cpm = price
        cpc = budget_usd / clicks if clicks > 0 else None

    conversions = clicks * cvr
    cpa = budget_usd / conversions if conversions > 0 else None
    frequency = impressions / potential_reach if potential_reach > 0 else None

    return {
        "budget_usd": budget_usd,
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "cpa_usd": cpa,
        "cpm_usd": cpm,
        "cpc_usd": cpc,
        "ctr": ctr,
        "cvr": cvr,
        "frequency": frequency,
    }


# ============================================================
# 3) بناء خطة كاملة
# ============================================================
def build_plan(
    *,
    platform_key: str,
    objective_key: str,
    geo_key: str,
    age_min: int,
    age_max: int,
    interests: list[str],
    keywords: list[str],
    budget: float,
    currency: str,
    pricing_model: str,
    price_usd: float,
    fx_rate: float | None = None,
) -> dict[str, Any]:
    """يجمع مدخلات الحملة ويحسب الجمهور والنتائج والسيناريوهات."""
    currency = currency if currency in CURRENCIES else "JOD"
    rate = float(fx_rate or CURRENCIES[currency]["per_usd"])
    rate = rate if rate > 0 else 1.0
    budget = max(0.0, float(budget))
    budget_usd = budget / rate

    platform = PLATFORMS[platform_key]
    objective = OBJECTIVES[objective_key]
    geo = GEOS[geo_key]

    audience = estimate_audience(
        platform_key, geo_key, age_min, age_max, interests, keywords
    )
    ctr, cvr = effective_rates(platform_key, objective_key)
    base = calculate_results(
        budget_usd, pricing_model, price_usd, ctr, cvr, audience["potential_reach"]
    )

    scenarios = []
    for _key, label_ar, label_en, ctr_mult, cvr_mult, cost_mult in SCENARIOS:
        result = calculate_results(
            budget_usd,
            pricing_model,
            max(0.0001, float(price_usd) * cost_mult),
            min(0.95, ctr * ctr_mult),
            min(0.95, cvr * cvr_mult),
            audience["potential_reach"],
        )
        result["label_ar"] = label_ar
        result["label_en"] = label_en
        result["cpa_display"] = result["cpa_usd"] * rate if result["cpa_usd"] is not None else None
        scenarios.append(result)

    allocation = [
        {
            "key": key,
            "label": label,
            "pct": pct,
            "amount": budget * pct,
        }
        for key, label, pct in BUDGET_ALLOCATION
    ]

    return {
        "input": {
            "platform_key": platform_key,
            "platform_ar": platform["label_ar"],
            "platform_en": platform["label_en"],
            "platform_icon": platform["icon"],
            "objective_key": objective_key,
            "objective_ar": objective["label_ar"],
            "objective_en": objective["label_en"],
            "objective_icon": objective["icon"],
            "geo_key": geo_key,
            "geo_ar": geo["label_ar"],
            "geo_en": geo["label_en"],
            "age_min": int(age_min),
            "age_max": int(age_max),
            "interests": list(interests),
            "keywords": list(keywords),
            "budget": budget,
            "currency": currency,
            "fx": rate,
            "pricing_model": pricing_model.upper(),
            "price_usd": float(price_usd),
            "budget_usd": budget_usd,
        },
        "audience": audience,
        "rates": {"ctr": ctr, "cvr": cvr},
        "results": base,
        "scenarios": scenarios,
        "allocation": allocation,
    }


# ============================================================
# 4) الرسوم البيانية
# ============================================================
def funnel_figure(plan: dict[str, Any]) -> go.Figure:
    """قمع النتائج: ظهور ← نقرات ← تحويلات."""
    res = plan["results"]
    labels = ["الظهور | Impressions", "النقرات | Clicks", "التحويلات | Conversions"]
    values = [res["impressions"], res["clicks"], res["conversions"]]
    fig = go.Figure(
        go.Funnel(
            y=labels,
            x=values,
            textinfo="value",
            texttemplate="%{value:,.0f}",
            marker={"color": ["#FF7A1A", "#FFB066", "#FFD1A3"]},
            connector={"line": {"color": "#2A2A30", "width": 1}},
        )
    )
    fig.update_layout(
        template=DARK_TEMPLATE,
        height=360,
        margin={"t": 30, "b": 10, "l": 10, "r": 10},
        showlegend=False,
        title={"text": "قمع النتائج المتوقعة | Expected Funnel", "font": {"size": 15}},
    )
    return fig


def allocation_figure(plan: dict[str, Any]) -> go.Figure:
    """توزيع الميزانية المقترح."""
    allocation = plan["allocation"]
    currency = plan["input"]["currency"]
    labels = [item["label"] for item in allocation]
    values = [item["amount"] for item in allocation]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker={"colors": ["#FF7A1A", "#FFB066", "#5A5A64"]},
            textinfo="label+percent",
            hovertemplate=(
                "%{label}<br>%{value:,.0f} " + currency + "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template=DARK_TEMPLATE,
        height=360,
        margin={"t": 30, "b": 10, "l": 10, "r": 10},
        showlegend=False,
        title={"text": "توزيع الميزانية | Budget Allocation", "font": {"size": 15}},
    )
    return fig


# ============================================================
# 5) التصدير
# ============================================================
def _recommendations(plan: dict[str, Any]) -> list[str]:
    res = plan["results"]
    audience = plan["audience"]
    items: list[str] = []

    freq = res["frequency"]
    if freq is not None and freq > 3:
        items.append(
            "متوسط التكرار مرتفع — وسّع الجمهور أو أضف تنويعات إبداعية لتجنب إجهاد الإعلان."
        )
    if res["ctr"] < 0.01:
        items.append(
            "نسبة النقر منخفضة — اختبر عنوانًا/فيديو أقوى وأضف دعوة إجراء واضحة."
        )
    if audience["interest_count"] + audience["keyword_count"] > 5:
        items.append(
            "الاستهداف ضيّق جدًا — قد يرتفع CPM؛ جرّب تقليل الاهتمامات أو الجمع بينها كمجموعات."
        )
    if plan["input"]["objective_key"] in ("conversions", "leads"):
        items.append(
            "فعّل تتبع التحويلات (Pixel / Conversions API) قبل الإطلاق لقياس CPA بدقة."
        )
    if plan["input"]["pricing_model"] == "CPM":
        items.append(
            "راقب CTR أسبوعيًا؛ إذا كان مرتفعًا ففكر في التحويل إلى CPC أو تحسين الاستهداف."
        )
    else:
        items.append(
            "راقب جودة النقرات؛ قد تجذب CPC المنخفض نقرات غير مؤهلة."
        )
    items.append(
        "ابدأ بـ 10% من الميزانية كاختبار لمدة 3–5 أيام، ثم أعد توزيعها على أفضل مجموعة."
    )
    return items


def plan_markdown(plan: dict[str, Any]) -> str:
    """تصدير خطة الحملة كملف Markdown عربي/إنجليزي."""
    inp = plan["input"]
    aud = plan["audience"]
    res = plan["results"]
    currency = inp["currency"]
    rate = inp["fx"]

    cpa_display = res["cpa_usd"] * rate if res["cpa_usd"] is not None else None
    lines: list[str] = [
        "# خطة الحملة الإعلانية | Campaign Plan",
        "",
        f"_تاريخ الإنشاء: {_dt.date.today().isoformat()}_",
        "",
        "## 🎯 مدخلات الحملة | Campaign Inputs",
        "",
        "| البند | القيمة |",
        "|---|---|",
        f"| المنصة | {inp['platform_icon']} {inp['platform_ar']} ({inp['platform_en']}) |",
        f"| الهدف | {inp['objective_icon']} {inp['objective_ar']} ({inp['objective_en']}) |",
        f"| النطاق الجغرافي | {inp['geo_ar']} ({inp['geo_en']}) |",
        f"| الفئة العمرية | {inp['age_min']}–{inp['age_max']} سنة |",
        f"| الاهتمامات | {', '.join(inp['interests']) if inp['interests'] else 'عام — بدون تضييق'} |",
        f"| الكلمات المفتاحية | {', '.join(inp['keywords']) if inp['keywords'] else '—'} |",
        f"| الميزانية | {format_money(inp['budget'], currency)} |",
        f"| نموذج التسعير | {inp['pricing_model']} — {format_money(inp['price_usd'], 'USD')} |",
        f"| سعر الصرف | 1 USD = {rate:,.3f} {currency} |",
        "",
        "## 👥 تقدير الجمهور | Audience Estimate",
        "",
        "| المرحلة | العدد | ملاحظة |",
        "|---|---:|---|",
    ]
    lines.extend(
        (
            f"| {step['المرحلة | Stage']} | {step['العدد | People']:,} | "
            f"{step['الملاحظة | Note']} |"
        )
        for step in aud["steps"]
    )
    lines += [
        "",
        f"**الوصول المحتمل | Potential Reach:** {aud['potential_reach']:,} شخص",
        "",
        "## 📈 النتائج المتوقعة | Expected Results",
        "",
        "| المؤشر | القيمة |",
        "|---|---:|",
        f"| الظهور | {format_int(res['impressions'])} |",
        f"| النقرات | {format_int(res['clicks'])} |",
        f"| التحويلات / العملاء المحتملون | {format_int(res['conversions'])} |",
        f"| CPM الفعلي | {format_money(res['cpm_usd'], 'USD')} |",
        f"| CPC الفعلي | {format_money(res['cpc_usd'], 'USD')} |",
        f"| CTR | {format_pct(res['ctr'])} |",
        f"| CVR | {format_pct(res['cvr'])} |",
        f"| CPA | {format_money(cpa_display, currency)} |",
        f"| متوسط التكرار | {format_int(res['frequency'])} |",
        "",
        "## 🔀 السيناريوهات | Scenarios",
        "",
        "| السيناريو | الظهور | النقرات | التحويلات | CPA |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in plan["scenarios"]:
        lines.append(
            f"| {item['label_ar']} / {item['label_en']} | "
            f"{format_int(item['impressions'])} | {format_int(item['clicks'])} | "
            f"{format_int(item['conversions'])} | {format_money(item['cpa_display'], currency)} |"
        )
    lines += [
        "",
        "## 💰 توزيع الميزانية المقترح | Suggested Allocation",
        "",
        "| البند | النسبة | المبلغ |",
        "|---|---:|---:|",
    ]
    for item in plan["allocation"]:
        lines.append(
            f"| {item['label']} | {item['pct'] * 100:.0f}% | "
            f"{format_money(item['amount'], currency)} |"
        )
    lines += ["", "## ✅ توصيات | Recommendations", ""]
    lines += [f"- {item}" for item in _recommendations(plan)]
    lines += [
        "",
        "---",
        "",
        (
            "> الأرقام تقديرية مبنية على متوسطات السوق وقد تختلف حسب جودة الإبداع، "
            "المنافسة، والموسم. راقب النتائج الفعلية وعدّل الميزانية أسبوعيًا."
        ),
    ]
    return "\n".join(lines)


def plan_csv(plan: dict[str, Any]) -> bytes:
    """تصدير ملخص الخطة كـ CSV بترميز UTF-8 مع BOM."""
    inp = plan["input"]
    aud = plan["audience"]
    res = plan["results"]
    currency = inp["currency"]
    rate = inp["fx"]
    cpa_display = res["cpa_usd"] * rate if res["cpa_usd"] is not None else None

    rows = [
        ("مدخلات | Input", "المنصة | Platform", f"{inp['platform_ar']} / {inp['platform_en']}"),
        ("مدخلات | Input", "الهدف | Objective", f"{inp['objective_ar']} / {inp['objective_en']}"),
        ("مدخلات | Input", "النطاق | Geo", f"{inp['geo_ar']} / {inp['geo_en']}"),
        ("مدخلات | Input", "العمر | Age", f"{inp['age_min']}–{inp['age_max']}"),
        ("مدخلات | Input", "الاهتمامات | Interests", "، ".join(inp["interests"]) or "عام"),
        ("مدخلات | Input", "الكلمات المفتاحية | Keywords", "، ".join(inp["keywords"]) or "—"),
        ("مدخلات | Input", "الميزانية | Budget", f"{inp['budget']:,.2f} {currency}"),
        ("مدخلات | Input", "نموذج التسعير | Pricing", inp["pricing_model"]),
        ("مدخلات | Input", "السعر | Price", f"{inp['price_usd']:,.2f} USD"),
        ("مدخلات | Input", "سعر الصرف | FX", f"1 USD = {rate:,.3f} {currency}"),
        ("جمهور | Audience", "الوصول المحتمل | Potential Reach", f"{aud['potential_reach']:,}"),
        ("نتائج | Results", "الظهور | Impressions", format_int(res["impressions"])),
        ("نتائج | Results", "النقرات | Clicks", format_int(res["clicks"])),
        ("نتائج | Results", "التحويلات | Conversions", format_int(res["conversions"])),
        ("نتائج | Results", "CPM (USD)", format_money(res["cpm_usd"], "USD")),
        ("نتائج | Results", "CPC (USD)", format_money(res["cpc_usd"], "USD")),
        ("نتائج | Results", "CTR", format_pct(res["ctr"])),
        ("نتائج | Results", "CVR", format_pct(res["cvr"])),
        ("نتائج | Results", "CPA", format_money(cpa_display, currency)),
        ("نتائج | Results", "التكرار | Frequency", format_int(res["frequency"])),
    ]
    rows.extend(
        (
            "سيناريو | Scenario",
            f"{item['label_ar']} / {item['label_en']}",
            (
                f"Impressions {format_int(item['impressions'])} · "
                f"Clicks {format_int(item['clicks'])} · "
                f"Conversions {format_int(item['conversions'])} · "
                f"CPA {format_money(item['cpa_display'], currency)}"
            ),
        )
        for item in plan["scenarios"]
    )
    rows.extend(
        (
            "توزيع | Allocation",
            item["label"],
            f"{item['pct'] * 100:.0f}% — {format_money(item['amount'], currency)}",
        )
        for item in plan["allocation"]
    )

    df = pd.DataFrame(rows, columns=["القسم | Section", "البند | Item", "القيمة | Value"])
    return df.to_csv(index=False).encode("utf-8-sig")
