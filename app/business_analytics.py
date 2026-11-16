"""
business_analytics.py
----------------------
محرك تحليل الأعمال: المبيعات، المصاريف، الربح، ومؤشرات السوق.

مصمَّم ليعمل مع أي ملف مبيعات/فواتير دون إعداد مسبق:
يكتشف الأعمدة تلقائيًا (التاريخ، المبلغ، التكلفة، العميل، الخدمة، الحالة)
ثم يحسب المؤشرات ويرسم التحليلات.

يعمل مع السوق المحلي (JOD، ضريبة 16%) والعالمي (عملات متعددة).

المطوّر: sari refai
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
# 0) تحويل الأعمدة النصية الرقمية
# ============================================================

CURRENCY_PATTERN = (
    r"(?i)\s*(jod|usd|eur|sar|aed|ils|egp|kwd|qar|omr|bhd|"
    r"د\.أ|دينار|ريال|درهم|شيكل|جنيه|\$|€|₪)\s*"
)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """يحوّل أعمدة نصية مثل "150 JOD" أو "1,200.50" إلى أرقام.

    يتجاهل الأعمدة النصية الحقيقية (الأسماء، التواريخ، الحالات).
    """
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_numeric_dtype(out[column]):
            continue
        cleaned = (
            out[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(CURRENCY_PATTERN, "", regex=True)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
        )
        converted = pd.to_numeric(cleaned, errors="coerce")
        if converted.notna().mean() >= 0.6:
            out[column] = converted
    return out


# ============================================================
# 1) اكتشاف الأعمدة تلقائيًا
# ============================================================

DATE_HINTS = ("date", "day", "time", "تاريخ", "يوم", "وقت")
REVENUE_HINTS = (
    "amount", "total", "price", "revenue", "sales", "income", "value", "paid",
    "مبلغ", "قيمة", "سعر", "إجمالي", "اجمالي", "مبيعات", "إيراد", "ايراد", "مدفوع",
)
EXPENSE_HINTS = (
    "expense", "cost", "spend", "purchase", "مصروف", "مصاريف", "تكلفة", "شراء",
)
PROFIT_HINTS = ("profit", "margin", "net", "ربح", "صافي", "هامش")
QTY_HINTS = ("qty", "quantity", "count", "units", "عدد", "كمية", "وحدات")
CLIENT_HINTS = ("client", "customer", "buyer", "عميل", "زبون", "اسم")
CATEGORY_HINTS = (
    "service", "category", "product", "item", "type", "خدمة", "نوع", "فئة", "منتج", "صنف",
)
STATUS_HINTS = ("status", "state", "payment", "حالة", "دفع")

PAID_WORDS = ("completed", "complete", "paid", "done", "success", "delivered", "مكتمل", "مدفوع", "منجز", "تم")
PENDING_WORDS = ("pending", "hold", "waiting", "processing", "معلق", "مؤجل", "قيد")
LOST_WORDS = ("cancel", "refund", "failed", "rejected", "ملغي", "الغاء", "إلغاء", "مرتجع", "مسترجع", "فاشل")


def _norm(value) -> str:
    return str(value).strip().lower()


def _has_hint(column, hints) -> bool:
    name = _norm(column)
    return any(hint in name for hint in hints)


def _find_column(df: pd.DataFrame, hints, exclude=()) -> str | None:
    for column in df.columns:
        if column in exclude:
            continue
        if _has_hint(column, hints):
            return column
    return None


def _find_numeric(df: pd.DataFrame, hints, exclude=()) -> str | None:
    for column in df.columns:
        if column in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        if _has_hint(column, hints):
            return column
    return None


def parse_dates(series: pd.Series, day_first: bool = True) -> pd.Series:
    """يحلّل التواريخ بصيغ مختلطة دون قلب الشهر واليوم.

    الصيغة ISO (2026-05-01) تُقرأ سنة-شهر-يوم،
    وغيرها (03/07/2026) تُقرأ حسب day_first.
    """
    text = series.astype(str).str.strip()
    iso_mask = text.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
    iso_mask = iso_mask.fillna(False)

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(
            text[iso_mask], errors="coerce", yearfirst=True
        )
    if (~iso_mask).any():
        parsed.loc[~iso_mask] = pd.to_datetime(
            text[~iso_mask], errors="coerce", dayfirst=day_first
        )
    return parsed


def _looks_like_date(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    parsed = parse_dates(sample)
    return parsed.notna().mean() >= 0.8


def detect_roles(df: pd.DataFrame) -> dict[str, str | None]:
    """يرجع قاموسًا يربط كل دور تحليلي باسم العمود المناسب."""
    roles: dict[str, str | None] = {}

    date_col = _find_column(df, DATE_HINTS)
    if date_col is None:
        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]) and _looks_like_date(df[column]):
                date_col = column
                break
    roles["date"] = date_col

    roles["expense"] = _find_numeric(df, EXPENSE_HINTS)
    roles["profit"] = _find_numeric(df, PROFIT_HINTS)
    roles["quantity"] = _find_numeric(df, QTY_HINTS, exclude={roles["expense"], roles["profit"]})

    revenue = _find_numeric(df, REVENUE_HINTS, exclude={roles["expense"], roles["profit"], roles["quantity"]})
    if revenue is None:
        # احتياطي: أول عمود رقمي موجب المعنى غير المصاريف/الربح/الكمية
        for column in df.columns:
            if column in {roles["expense"], roles["profit"], roles["quantity"]}:
                continue
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
            if (df[column].dropna() >= 0).mean() >= 0.8:
                revenue = column
                break
    roles["revenue"] = revenue

    roles["client"] = _find_column(df, CLIENT_HINTS, exclude={roles["date"]})
    roles["category"] = _find_column(df, CATEGORY_HINTS, exclude={roles["date"], roles["client"]})
    roles["status"] = _find_column(df, STATUS_HINTS, exclude={roles["date"], roles["client"], roles["category"]})

    # لا نعتبر عمود النص "عميل" إذا كان نصًا حرًا طويلًا جدًا لكل صف
    return roles


# ============================================================
# 2) تجهيز البيانات
# ============================================================

def normalize_status(value) -> str:
    """يصنّف الحالة إلى: paid / pending / lost / other."""
    text = _norm(value)
    if any(word in text for word in LOST_WORDS):
        return "lost"
    if any(word in text for word in PENDING_WORDS):
        return "pending"
    if any(word in text for word in PAID_WORDS):
        return "paid"
    return "other"


def prepare(df: pd.DataFrame, roles: dict, day_first: bool = True) -> pd.DataFrame:
    """ينشئ إطارًا موحدًا بأعمدة داخلية: _date, _revenue, _expense, _profit, _status."""
    out = pd.DataFrame(index=df.index)

    if roles.get("date"):
        out["_date"] = parse_dates(df[roles["date"]], day_first=day_first)
    else:
        out["_date"] = pd.NaT

    def numeric(col):
        if not col:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    out["_revenue"] = numeric(roles.get("revenue"))
    out["_expense"] = numeric(roles.get("expense"))

    if roles.get("profit"):
        out["_profit"] = numeric(roles["profit"])
    else:
        out["_profit"] = out["_revenue"] - out["_expense"]

    out["_qty"] = numeric(roles.get("quantity")).replace(0, 1.0) if roles.get("quantity") else 1.0

    if roles.get("status"):
        out["_status"] = df[roles["status"]].map(normalize_status)
    else:
        out["_status"] = "paid"

    out["_client"] = df[roles["client"]].astype(str) if roles.get("client") else "غير محدد"
    out["_category"] = df[roles["category"]].astype(str) if roles.get("category") else "غير مصنّف"

    return out


def daily_summary(prepared: pd.DataFrame) -> pd.DataFrame:
    """تجميع يومي: الإيراد، المصروف، الربح، عدد الفواتير."""
    data = prepared.dropna(subset=["_date"]).copy()
    if data.empty:
        return pd.DataFrame(columns=["date", "revenue", "expense", "profit", "orders"])

    data["date"] = data["_date"].dt.normalize()
    grouped = data.groupby("date").agg(
        revenue=("_revenue", "sum"),
        expense=("_expense", "sum"),
        profit=("_profit", "sum"),
        orders=("_revenue", "size"),
    )
    grouped["margin"] = np.where(grouped["revenue"] != 0, grouped["profit"] / grouped["revenue"], 0.0)
    return grouped.reset_index()


# ============================================================
# 3) المؤشرات الرئيسية
# ============================================================

def _growth(current: float, previous: float) -> float | None:
    if previous is None or previous == 0 or np.isnan(previous):
        return None
    return (current - previous) / abs(previous)


def compute_kpis(prepared: pd.DataFrame, vat_rate: float = 0.0, currency: str = "JOD") -> dict:
    revenue = float(prepared["_revenue"].sum())
    expense = float(prepared["_expense"].sum())
    profit = float(prepared["_profit"].sum())
    orders = len(prepared)
    avg_ticket = revenue / orders if orders else 0.0

    paid_mask = prepared["_status"] == "paid"
    lost_mask = prepared["_status"] == "lost"
    pending_mask = prepared["_status"] == "pending"

    collected = float(prepared.loc[paid_mask, "_revenue"].sum())
    lost_value = float(prepared.loc[lost_mask, "_revenue"].sum())
    pending_value = float(prepared.loc[pending_mask, "_revenue"].sum())

    # النمو: مقارنة آخر فترة بفترة مساوية قبلها
    growth = None
    period_days = 0
    dated = prepared.dropna(subset=["_date"])
    if not dated.empty:
        last_day = dated["_date"].max()
        first_day = dated["_date"].min()
        span = (last_day - first_day).days + 1
        period_days = max(1, min(30, span // 2))
        current_start = last_day - pd.Timedelta(days=period_days - 1)
        previous_start = current_start - pd.Timedelta(days=period_days)

        current_rev = dated.loc[dated["_date"] >= current_start, "_revenue"].sum()
        previous_rev = dated.loc[
            (dated["_date"] >= previous_start) & (dated["_date"] < current_start), "_revenue"
        ].sum()
        growth = _growth(float(current_rev), float(previous_rev))

    return {
        "currency": currency,
        "revenue": revenue,
        "expense": expense,
        "profit": profit,
        "margin": (profit / revenue) if revenue else 0.0,
        "orders": orders,
        "avg_ticket": avg_ticket,
        "collected": collected,
        "pending_value": pending_value,
        "lost_value": lost_value,
        "collection_rate": (collected / revenue) if revenue else 0.0,
        "growth": growth,
        "growth_period_days": period_days,
        "vat_rate": vat_rate,
        "revenue_ex_vat": revenue / (1 + vat_rate) if vat_rate > -1 else revenue,
        "vat_amount": revenue - (revenue / (1 + vat_rate)) if vat_rate > -1 else 0.0,
        "profit_ex_vat": profit - (revenue - (revenue / (1 + vat_rate))) if vat_rate > -1 else profit,
        "expense_ratio": (expense / revenue) if revenue else 0.0,
    }


def concentration(prepared: pd.DataFrame, by: str = "_client", top: int = 1) -> float:
    """حصة أكبر عميل/خدمة من إجمالي الإيراد (مؤشر مخاطر التركّز)."""
    totals = prepared.groupby(by)["_revenue"].sum().sort_values(ascending=False)
    if totals.empty or totals.sum() == 0:
        return 0.0
    return float(totals.head(top).sum() / totals.sum())


def pareto(prepared: pd.DataFrame, by: str = "_client", top: int = 10) -> pd.DataFrame:
    """تحليل ABC / باريتو: أعلى العناصر وحصتها التراكمية."""
    totals = (
        prepared.groupby(by)
        .agg(revenue=("_revenue", "sum"), orders=("_revenue", "size"))
        .sort_values("revenue", ascending=False)
        .head(top)
        .reset_index()
        .rename(columns={by: "name"})
    )
    if totals.empty:
        return totals

    grand = float(prepared["_revenue"].sum())
    totals["share"] = totals["revenue"] / grand if grand else 0.0
    totals["cum_share"] = totals["share"].cumsum()
    totals["class"] = np.where(
        totals["cum_share"] <= 0.8, "A", np.where(totals["cum_share"] <= 0.95, "B", "C")
    )
    return totals


# ============================================================
# 4) كشف الانحراف في الإيراد اليومي
# ============================================================

def flag_revenue_anomalies(daily: pd.DataFrame, column: str = "revenue", threshold: float = 2.5) -> pd.DataFrame:
    """يضيف عمودي is_anomaly و z_score باستخدام Z-score قوي (median / MAD)."""
    result = daily.copy()
    if result.empty or column not in result:
        result["z_score"] = 0.0
        result["is_anomaly"] = False
        return result

    values = result[column].astype(float)
    median = values.median()
    mad = (values - median).abs().median()
    scale = mad * 1.4826 if mad > 0 else values.std(ddof=0)

    if not scale or np.isnan(scale) or scale == 0:
        result["z_score"] = 0.0
        result["is_anomaly"] = False
        return result

    result["z_score"] = (values - median) / scale
    result["is_anomaly"] = result["z_score"].abs() >= threshold
    return result


# ============================================================
# 5) توقّع الإيراد
# ============================================================

def forecast_revenue(daily: pd.DataFrame, periods: int = 14, column: str = "revenue") -> pd.DataFrame:
    """توقّع بسيط: اتجاه خطي + موسمية أسبوعية.

    يرجع إطارًا يحتوي date, forecast, lower, upper.
    """
    if daily.empty or column not in daily or len(daily) < 3:
        return pd.DataFrame(columns=["date", "forecast", "lower", "upper"])

    series = daily.set_index("date")[column].astype(float).asfreq("D").interpolate()
    y = series.to_numpy()
    x = np.arange(len(y), dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    trend = slope * x + intercept
    residual = y - trend

    # موسمية أسبوعية: متوسط البواقي لكل يوم في الأسبوع
    weekday = series.index.dayofweek
    seasonal = pd.Series(residual, index=weekday).groupby(level=0).mean()

    last_date = series.index.max()
    future_index = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")
    future_x = np.arange(len(y), len(y) + periods, dtype=float)

    forecast = slope * future_x + intercept
    forecast = forecast + np.array([seasonal.get(d, 0.0) for d in future_index.dayofweek])

    sigma = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
    return pd.DataFrame(
        {
            "date": future_index,
            "forecast": np.clip(forecast, 0, None),
            "lower": np.clip(forecast - 1.96 * sigma, 0, None),
            "upper": np.clip(forecast + 1.96 * sigma, 0, None),
        }
    )


# ============================================================
# 6) مؤشرات السوق
# ============================================================

CURRENCIES = {
    "JOD": {"name_ar": "دينار أردني", "vat": 0.16, "symbol": "د.أ"},
    "USD": {"name_ar": "دولار أمريكي", "vat": 0.0, "symbol": "$"},
    "EUR": {"name_ar": "يورو", "vat": 0.20, "symbol": "€"},
    "SAR": {"name_ar": "ريال سعودي", "vat": 0.15, "symbol": "ر.س"},
    "AED": {"name_ar": "درهم إماراتي", "vat": 0.05, "symbol": "د.إ"},
    "ILS": {"name_ar": "شيكل", "vat": 0.17, "symbol": "₪"},
}


def market_score(kpis: dict, target_margin: float = 0.25) -> dict:
    """تقييم صحة الأعمال (0-100) مقابل أهداف السوق.

    يعتمد على: هامش الربح، نسبة المصاريف، نسبة التحصيل، ومخاطر التركّز.
    """
    margin = kpis.get("margin", 0.0)
    expense_ratio = kpis.get("expense_ratio", 0.0)
    collection = kpis.get("collection_rate", 0.0)
    top_client_share = kpis.get("top_client_share", 0.0)

    margin_score = float(np.clip(margin / target_margin, 0, 1) * 35) if target_margin > 0 else 0.0
    expense_score = float(np.clip(1 - expense_ratio / 0.9, 0, 1) * 25)
    collection_score = float(np.clip(collection, 0, 1) * 25)
    risk_score = float(np.clip(1 - top_client_share / 0.5, 0, 1) * 15)

    total = margin_score + expense_score + collection_score + risk_score
    if total >= 80:
        label = "ممتاز | Excellent"
    elif total >= 60:
        label = "جيد | Good"
    elif total >= 40:
        label = "متوسط | Fair"
    else:
        label = "يحتاج تحسين | Needs work"

    return {
        "score": round(total, 1),
        "label": label,
        "margin_score": round(margin_score, 1),
        "expense_score": round(expense_score, 1),
        "collection_score": round(collection_score, 1),
        "risk_score": round(risk_score, 1),
    }


def break_even(daily: pd.DataFrame) -> dict:
    """نقطة التعادل: الإيراد اليومي المطلوب لتغطية المصاريف."""
    if daily.empty:
        return {"daily": 0.0, "monthly": 0.0}
    avg_expense = float(daily["expense"].mean())
    avg_margin = float(daily["margin"].mean()) if "margin" in daily else 0.0
    if avg_margin <= 0:
        return {"daily": avg_expense, "monthly": avg_expense * 30, "note": "الهامش غير كافٍ"}
    daily_break_even = avg_expense / avg_margin
    return {"daily": daily_break_even, "monthly": daily_break_even * 30}


# ============================================================
# 7) تحليلات إضافية للسوق
# ============================================================

WEEKDAY_NAMES = {
    0: "الاثنين | Mon",
    1: "الثلاثاء | Tue",
    2: "الأربعاء | Wed",
    3: "الخميس | Thu",
    4: "الجمعة | Fri",
    5: "السبت | Sat",
    6: "الأحد | Sun",
}


def weekday_profile(daily: pd.DataFrame) -> pd.DataFrame:
    """متوسط الإيراد والربح حسب يوم الأسبوع (موسمية أسبوعية)."""
    if daily.empty:
        return pd.DataFrame(columns=["weekday", "revenue", "profit", "orders"])
    data = daily.copy()
    data["weekday"] = data["date"].dt.dayofweek
    grouped = data.groupby("weekday").agg(
        revenue=("revenue", "mean"),
        profit=("profit", "mean"),
        orders=("orders", "mean"),
    )
    grouped["name"] = [WEEKDAY_NAMES.get(int(d), str(d)) for d in grouped.index]
    return grouped.reset_index()


def monthly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """ملخص شهري: الإيراد، المصروف، الربح، الهامش، عدد الفواتير."""
    if daily.empty:
        return pd.DataFrame(columns=["month", "revenue", "expense", "profit", "margin", "orders"])
    data = daily.set_index("date").resample("MS").agg(
        revenue=("revenue", "sum"),
        expense=("expense", "sum"),
        profit=("profit", "sum"),
        orders=("orders", "sum"),
    ).reset_index()
    data["margin"] = np.where(data["revenue"] != 0, data["profit"] / data["revenue"], 0.0)
    data["label"] = data["date"].dt.strftime("%Y-%m")
    return data


def status_breakdown(prepared: pd.DataFrame) -> pd.DataFrame:
    """توزيع الفواتير حسب الحالة (مدفوع/معلق/ملغي) بالقيمة والعدد."""
    if prepared.empty:
        return pd.DataFrame(columns=["status", "value", "orders"])
    grouped = prepared.groupby("_status").agg(
        value=("_revenue", "sum"),
        orders=("_revenue", "size"),
    )
    names = {
        "paid": "مدفوع | Paid",
        "pending": "معلق | Pending",
        "lost": "ملغي/مرتجع | Lost",
        "other": "أخرى | Other",
    }
    grouped["label"] = [names.get(s, s) for s in grouped.index]
    return grouped.reset_index()


def analyze(df: pd.DataFrame, vat_rate: float = 0.0, currency: str = "JOD",
            target_margin: float = 0.25, forecast_days: int = 14) -> dict:
    """يحلل أي ملف مبيعات ويعيد كل المؤشرات والجداول جاهزة للعرض."""
    working = coerce_numeric_columns(df)
    roles = detect_roles(working)
    prepared = prepare(working, roles)
    daily = flag_revenue_anomalies(daily_summary(prepared))

    kpis = compute_kpis(prepared, vat_rate=vat_rate, currency=currency)
    kpis["top_client_share"] = concentration(prepared, "_client")
    kpis["top_service_share"] = concentration(prepared, "_category")

    return {
        "roles": roles,
        "prepared": prepared,
        "daily": daily,
        "monthly": monthly_summary(daily),
        "weekday": weekday_profile(daily),
        "status": status_breakdown(prepared),
        "clients": pareto(prepared, "_client", top=10),
        "services": pareto(prepared, "_category", top=10),
        "kpis": kpis,
        "score": market_score(kpis, target_margin=target_margin),
        "break_even": break_even(daily),
        "forecast": forecast_revenue(daily, periods=forecast_days),
        "has_expense": bool(roles.get("expense")),
        "has_date": bool(roles.get("date")),
    }
