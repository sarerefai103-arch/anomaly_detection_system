"""
predictive_analytics.py
-----------------------
التحليل التنبؤي المدعوم بالذكاء الاصطناعي (Predictive Analytics).

يوفّر:
- توقع المبيعات/الإيرادات المستقبلية (اتجاه + موسمية أسبوعية + فترات ثقة).
- توقع الطلب على المنتجات/الخدمات.
- توقع انخفاض الأرباح (مستوى خطر، نقطة تحوّل/هبوط متوقعة).
- توقع سلوك العملاء (RFM، مخاطر فقدان العميل، إشارات قيمة).
- رؤى ذكية اختيارية بالذكاء الاصطناعي مع بديل محلي.

يعمل مع مخرجات `business_analytics.analyze()`.
المطوّر: sari refai
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import urllib.error
from typing import Any

import numpy as np
import pandas as pd

from ai_report import DEFAULT_MODEL, MODELS

PREDICTIVE_SYSTEM = (
    "أنت خبير في التحليل التنبؤي وذكاء الأعمال (Predictive Analytics / Business AI). "
    "تكتب رؤى تنفيذية دقيقة ومهنية بالعربية والإنجليزية بالاعتماد حصريًا على الأرقام المعطاة. "
    "لا تخترع بيانات. اذكر بوضوح أن التوقعات تقديرية وتعتمد على الأنماط التاريخية."
)

_WORKER_SCRIPT = r"""
import json, sys, urllib.request, urllib.error
cfg = json.load(sys.stdin)
url = ("https://generativelanguage.googleapis.com/v1beta/models/"
       + cfg["model"] + ":generateContent?key=" + cfg["key"])
body = json.dumps({
    "systemInstruction": {"parts": [{"text": cfg["system"]}]},
    "contents": [{"role": "user", "parts": [{"text": cfg["prompt"]}]}],
    "generationConfig": {"temperature": 0.3, "topP": 0.95, "maxOutputTokens": 4096},
}).encode("utf-8")
request = urllib.request.Request(url, data=body,
    headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(request, timeout=cfg.get("http_timeout", 30)) as r:
        payload = json.load(r)
    parts = (((payload.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
    text = "\n".join(p.get("text","") for p in parts).strip()
    if not text: raise RuntimeError("empty response")
    print(json.dumps({"ok": True, "text": text}))
except urllib.error.HTTPError as e:
    print(json.dumps({"ok": False, "code": e.code, "error": "HTTP %s" % e.code}))
except Exception as e:
    print(json.dumps({"ok": False, "code": None, "error": type(e).__name__+": "+str(e)[:180]}))
"""


# ============================================================
# أدوات مساعدة
# ============================================================
def _num(v: Any) -> float | None:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(n) or math.isinf(n)) else round(n, 4)


def _pct(v: float | None, d: int = 1) -> str:
    return "—" if v is None else f"{v:.{d}f}%"


def _money(v: float | None, cur: str) -> str:
    return "—" if v is None else f"{cur} {v:,.0f}"


def _safe_df(df: pd.DataFrame | None) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


# ============================================================
# 1) توقع السلاسل الزمنية (إيراد/ربح/مصاريف)
# ============================================================
def forecast_series(daily: pd.DataFrame, column: str, periods: int = 30) -> pd.DataFrame:
    """اتجاه خطي + موسمية أسبوعية + فاصل ثقة 95%."""
    if daily.empty or column not in daily.columns or len(daily) < 4:
        return pd.DataFrame(columns=["date", "forecast", "lower", "upper", "metric"])

    series = daily.set_index("date")[column].astype(float).asfreq("D").interpolate(limit_direction="both")
    y = series.to_numpy(dtype=float)
    n = len(y)

    last = series.index.max()
    future = pd.date_range(last + pd.Timedelta(days=1), periods=periods, freq="D")

    if n < 7:
        # سلسلة قصيرة جدًا: الانحدار الخطي يعطي قيمًا متطرفة، لذا نستخدم
        # متوسط آخر الأيام كتوقّع مستقر بدلًا من مدّ الاتجاه إلى 14+ يومًا.
        level = float(np.mean(y[-min(n, 5):]))
        fc = np.full(periods, level, dtype=float)
        residual = y - level
    else:
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        trend = slope * x + intercept
        residual = y - trend
        weekday = series.index.dayofweek
        seasonal = pd.Series(residual, index=weekday).groupby(level=0).mean()
        fx = np.arange(n, n + periods, dtype=float)
        fc = slope * fx + intercept + np.array([seasonal.get(d, 0.0) for d in future.dayofweek])
    sigma = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
    # توسيع طفيف للفاصل مع الأفق
    horizon = np.sqrt(1 + np.arange(1, periods + 1) / max(1, len(y)))
    band = 1.96 * sigma * horizon
    return pd.DataFrame({
        "date": future,
        "forecast": np.clip(fc, 0, None) if column != "profit" else fc,
        "lower": (np.clip(fc - band, 0, None) if column != "profit" else fc - band),
        "upper": (np.clip(fc + band, 0, None) if column != "profit" else fc + band),
        "metric": column,
    })


def forecast_summary(daily: pd.DataFrame, periods: int, currency: str) -> dict:
    rev = forecast_series(daily, "revenue", periods)
    exp = forecast_series(daily, "expense", periods) if "expense" in daily.columns else pd.DataFrame()
    prof = forecast_series(daily, "profit", periods) if "profit" in daily.columns else pd.DataFrame()

    def _sum(df: pd.DataFrame) -> float | None:
        return None if df.empty else _num(df["forecast"].sum())

    def _avg(df: pd.DataFrame) -> float | None:
        return None if df.empty else _num(df["forecast"].mean())

    # نمو متوقع مقارنة بآخر فترة مماثلة
    def _growth(df: pd.DataFrame, col: str) -> float | None:
        if df.empty or daily.empty or col not in daily.columns or len(daily) < periods:
            return None
        base = float(daily[col].tail(periods).sum())
        if base == 0:
            return None
        return _num((float(df["forecast"].sum()) - base) / abs(base) * 100)

    return {
        "periods": periods,
        "currency": currency,
        "revenue": {"total": _sum(rev), "avg_daily": _avg(rev), "growth_pct": _growth(rev, "revenue"), "df": rev},
        "expense": {"total": _sum(exp), "avg_daily": _avg(exp), "growth_pct": _growth(exp, "expense"), "df": exp},
        "profit": {"total": _sum(prof), "avg_daily": _avg(prof), "growth_pct": _growth(prof, "profit"), "df": prof},
    }


# ============================================================
# 2) توقع الطلب على المنتجات/الخدمات
# ============================================================
def forecast_demand(result: dict, periods: int = 30) -> pd.DataFrame:
    """توقّع الطلب (عدد الفواتير/الإيراد) لأعلى المنتجات/الخدمات."""
    prepared = _safe_df(result.get("prepared"))
    services = _safe_df(result.get("services"))
    if prepared.empty or services.empty or "_category" not in prepared.columns:
        return pd.DataFrame()

    if "_date" not in prepared.columns:
        return pd.DataFrame()

    top_names = services.head(6)["name"].astype(str).tolist()
    rows: list[dict] = []

    for name in top_names:
        subset = prepared[prepared["_category"].astype(str) == name].copy()
        if subset.empty:
            continue
        daily_demand = subset.groupby("_date").size().rename("orders").reset_index()
        daily_demand = daily_demand.rename(columns={"_date": "date"})
        # اتجاه بسيط للطلب اليومي
        y = daily_demand["orders"].astype(float).to_numpy()
        avg_orders = float(y.mean()) if len(y) else 0.0
        if len(y) >= 3:
            x = np.arange(len(y), dtype=float)
            slope, _ = np.polyfit(x, y, 1)
            trend_per_day = float(slope)
        else:
            trend_per_day = 0.0

        projected_orders = max(0.0, avg_orders * periods + trend_per_day * periods * 0.5)
        revenue = float(subset["_revenue"].sum()) if "_revenue" in subset.columns else 0.0
        avg_ticket = revenue / max(1, len(subset))
        projected_revenue = projected_orders * avg_ticket
        share = float(services.loc[services["name"].astype(str) == name, "share"].iloc[0]) * 100

        if trend_per_day > 0.02:
            trend_label = "صاعد | Rising"
            trend_icon = "📈"
        elif trend_per_day < -0.02:
            trend_label = "هابط | Falling"
            trend_icon = "📉"
        else:
            trend_label = "مستقر | Stable"
            trend_icon = "➡️"

        rows.append({
            "service": name,
            "historical_orders": len(subset),
            "avg_daily_orders": round(avg_orders, 2),
            "projected_orders": round(projected_orders, 1),
            "projected_revenue": round(projected_revenue, 2),
            "share_pct": round(share, 1),
            "trend": trend_label,
            "trend_icon": trend_icon,
            "trend_slope": round(trend_per_day, 4),
        })

    return pd.DataFrame(rows)


# ============================================================
# 3) توقع انخفاض الأرباح
# ============================================================
def predict_profit_decline(daily: pd.DataFrame, periods: int = 30) -> dict:
    """يقيّم خطر انخفاض الأرباح ويتوقّع نقطة تحوّل إن وجدت."""
    out: dict[str, Any] = {
        "available": False, "risk": "غير متاح | N/A", "risk_level": "unknown",
        "slope_daily": None, "recent_margin_pct": None, "projected_profit": None,
        "days_to_negative": None, "signals": [], "note": "",
    }
    if daily.empty or "profit" not in daily.columns or len(daily) < 5:
        out["note"] = "بيانات غير كافية (يلزم عمود ربح و5+ أيام)."
        return out

    s = daily.set_index("date")["profit"].astype(float).asfreq("D").interpolate(limit_direction="both")
    y = s.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    sigma = float(np.std(y - (slope * x + intercept), ddof=1)) if len(y) > 1 else 0.0

    recent_n = min(14, max(5, len(y) // 3))
    recent = y[-recent_n:]
    rx = np.arange(len(recent), dtype=float)
    r_slope, r_intercept = np.polyfit(rx, recent, 1)
    recent_avg = float(np.mean(recent))
    recent_margin = None
    if "revenue" in daily.columns:
        rev_recent = float(daily["revenue"].tail(recent_n).sum())
        if rev_recent > 0:
            recent_margin = _num(float(daily["profit"].tail(recent_n).sum()) / rev_recent * 100)

    # توقع الربح في نهاية الأفق
    proj_end = r_slope * (len(recent) - 1 + periods) + r_intercept
    proj_avg = float(np.mean([r_slope * (len(recent) - 1 + i) + r_intercept for i in range(1, periods + 1)]))

    # أيام حتى الربح السلبي (تقريبي) إذا الاتجاه هابط
    days_to_neg = None
    if r_slope < -1e-6 and recent_avg > 0:
        # من آخر نقطة حديثة
        last_val = float(r_slope * (len(recent) - 1) + r_intercept)
        if last_val > 0:
            days_to_neg = math.ceil(last_val / abs(r_slope))
            if days_to_neg > 3650:
                days_to_neg = None

    signals: list[str] = []
    risk_score = 0
    if r_slope < -5:
        signals.append("انحدار حاد في الربح خلال الفترة الأخيرة | Sharp recent profit decline")
        risk_score += 3
    elif r_slope < 0:
        signals.append("اتجاه الربح الأخير هابط | Recent profit trend is declining")
        risk_score += 2
    else:
        signals.append("اتجاه الربح الأخير مستقر/صاعد | Recent profit trend is stable/rising")

    if recent_avg < 0:
        signals.append("متوسط الربح الأخير سالب | Recent average profit is negative")
        risk_score += 3
    elif proj_end < 0:
        signals.append("المتوقع يشير لربح سلبي ضمن الأفق | Forecast may turn negative within horizon")
        risk_score += 2

    if recent_margin is not None and recent_margin < 10:
        signals.append(f"هامش ربحي حديث منخفض ({recent_margin:.1f}%) | Low recent margin")
        risk_score += 1

    volatility = (sigma / (abs(float(np.mean(y))) + 1e-9)) if sigma else 0
    if volatility > 1.2:
        signals.append("تذبذب عالٍ في الربح | High profit volatility")
        risk_score += 1

    if risk_score >= 5:
        risk, level = "مرتفع | High", "high"
    elif risk_score >= 3:
        risk, level = "متوسط | Medium", "medium"
    elif risk_score >= 1:
        risk, level = "منخفض | Low", "low"
    else:
        risk, level = "جيد | Healthy", "healthy"

    note = "تقدير مبني على الاتجاه الخطي للفترة الأخيرة؛ قد تتغير النتائج بعوامل موسمية/تشغيلية."
    if days_to_neg is not None and days_to_neg <= periods:
        note = f"⚠️ المؤشر يقدّر احتمال وصول الربح لمنطقة حرجة خلال ~{days_to_neg} يومًا إذا استمر الاتجاه."

    out.update({
        "available": True, "risk": risk, "risk_level": level,
        "slope_daily": _num(r_slope), "recent_margin_pct": recent_margin,
        "projected_profit": _num(proj_avg), "days_to_negative": days_to_neg,
        "signals": signals, "note": note, "volatility": _num(volatility),
    })
    return out


# ============================================================
# 4) توقع سلوك العملاء (RFM + مخاطر فقدان)
# ============================================================
def predict_customer_behavior(result: dict) -> dict:
    """تحليل RFM وتوقع سلوك/مخاطر فقدان العملاء."""
    prepared = _safe_df(result.get("prepared"))
    out: dict[str, Any] = {
        "available": False, "segments": pd.DataFrame(), "summary": {}, "signals": [],
        "note": "لا تتوفر بيانات عملاء كافية (يلزم أعمدة عميل وتاريخ وقيمة).",
    }
    if prepared.empty or "_client" not in prepared.columns or "_date" not in prepared.columns:
        return out

    df = prepared[["_client", "_date", "_revenue"]].dropna(subset=["_client", "_date"]).copy()
    df["_client"] = df["_client"].astype(str).str.strip()
    df = df[df["_client"].ne("") & ~df["_client"].str.lower().eq("nan")]
    if df.empty or df["_client"].nunique() < 2:
        return out

    ref_date = pd.to_datetime(df["_date"]).max()
    agg = df.groupby("_client").agg(
        recency_days=("_date", lambda x: (ref_date - pd.to_datetime(x).max()).days),
        frequency=("_date", "count"),
        monetary=("_revenue", "sum"),
        last_date=("_date", "max"),
    ).reset_index()

    # درجات RFM (1-5)
    def _score(series: pd.Series, reverse: bool = False) -> pd.Series:
        try:
            r = pd.qcut(series.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        except Exception:
            r = pd.Series([3] * len(series), index=series.index)
        return (6 - r) if reverse else r

    agg["R"] = _score(agg["recency_days"], reverse=True)
    agg["F"] = _score(agg["frequency"])
    agg["M"] = _score(agg["monetary"])
    agg["rfm_score"] = agg["R"] + agg["F"] + agg["M"]

    def _segment(row: pd.Series) -> tuple[str, str, str]:
        s = int(row["rfm_score"])
        r, f = int(row["R"]), int(row["F"])
        if s >= 13 and f >= 4:
            return "أبطال | Champions", "low", "🟢"
        if s >= 10 and r >= 4:
            return "أوفياء | Loyal", "low", "🟢"
        if r >= 4 and f <= 2:
            return "جدد/واعدون | New/Promising", "medium", "🟡"
        if r <= 2 and f >= 3:
            return "معرّضون للفقدان | At Risk", "high", "🟠"
        if r <= 2 and f <= 2:
            return "خاملون/مفقودون | Dormant/Lost", "high", "🔴"
        if s >= 8:
            return "بحاجة اهتمام | Need Attention", "medium", "🟡"
        return "غير نشط | Inactive", "high", "🔴"

    seg = agg.apply(_segment, axis=1, result_type="expand")
    agg["segment"] = seg[0]
    agg["churn_risk"] = seg[1]
    agg["icon"] = seg[2]
    agg = agg.sort_values(["monetary", "frequency"], ascending=False)

    total_customers = len(agg)
    at_risk = int(agg["churn_risk"].eq("high").sum())
    top_value_share = float(agg.head(max(1, total_customers // 5))["monetary"].sum()) / max(1e-9, float(agg["monetary"].sum())) * 100

    seg_counts = agg["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "customers"]

    signals = []
    if at_risk / max(1, total_customers) > 0.35:
        signals.append(f"نسبة مرتفعة من العملاء معرّضين للفقدان ({at_risk}/{total_customers}) | High churn-risk share")
    if top_value_share > 60:
        signals.append(f"تركّز القيمة: أعلى 20% من العملاء يولّدون {top_value_share:.1f}% | Revenue concentration risk")
    if int(agg["segment"].str.contains("Champions").sum()) > 0:
        signals.append("وجود عملاء أبطال — فرصة لبرامج ولاء/Upsell | Champions exist — loyalty/upsell opportunity")
    if not signals:
        signals.append("توزّع العملاء متوازن نسبيًا؛ حافظ على المتابعة الدورية | Balanced distribution; maintain follow-up")

    summary = {
        "total_customers": total_customers,
        "at_risk_customers": at_risk,
        "at_risk_pct": _num(at_risk / max(1, total_customers) * 100),
        "top20_value_share_pct": _num(top_value_share),
        "avg_monetary": _num(agg["monetary"].mean()),
        "avg_frequency": _num(agg["frequency"].mean()),
    }
    out.update({
        "available": True, "segments": agg, "segment_counts": seg_counts,
        "summary": summary, "signals": signals,
        "note": "تصنيف RFM تقديري مبني على الحداثة والتكرار والقيمة النقدية.",
    })
    return out


# ============================================================
# 5) محرك موحّد + سياق AI
# ============================================================
def run_predictive(result: dict, market: dict) -> dict:
    daily = _safe_df(result.get("daily"))
    periods = int(market.get("forecast_days", 30) or 30)
    periods = max(7, min(90, periods))
    currency = str(market.get("currency", "JOD"))
    return {
        "periods": periods,
        "currency": currency,
        "sales": forecast_summary(daily, periods, currency),
        "demand": forecast_demand(result, periods),
        "profit_decline": predict_profit_decline(daily, periods),
        "customers": predict_customer_behavior(result),
    }


def build_predictive_context(pred: dict, result: dict, market: dict) -> dict:
    sales = pred["sales"]
    rev = sales["revenue"]
    prof = sales["profit"]
    decline = pred["profit_decline"]
    cust = pred["customers"]
    demand = pred["demand"]
    return {
        "currency": pred["currency"],
        "horizon_days": pred["periods"],
        "market": {k: market.get(k) for k in ("currency", "vat_rate", "target_margin")},
        "revenue_forecast": {k: rev.get(k) for k in ("total", "avg_daily", "growth_pct")},
        "profit_forecast": {k: prof.get(k) for k in ("total", "avg_daily", "growth_pct")},
        "profit_risk": {k: decline.get(k) for k in ("risk", "risk_level", "slope_daily", "projected_profit", "days_to_negative", "signals", "recent_margin_pct")},
        "top_demand": demand.head(5).to_dict(orient="records") if not demand.empty else [],
        "customer_behavior": {
            "available": cust["available"],
            "summary": cust.get("summary", {}),
            "signals": cust.get("signals", []),
            "top_segments": cust.get("segment_counts", pd.DataFrame()).head(5).to_dict(orient="records") if cust["available"] else [],
        },
        "kpis": {k: _num(result.get("kpis", {}).get(k)) for k in ("revenue", "profit", "margin", "orders", "growth")},
    }


def _build_prompt(ctx: dict) -> str:
    return (
        "أنشئ تقرير تحليل تنبؤي تنفيذي (Predictive Analytics) بناءً على JSON التالي فقط.\n"
        "اكتب بالعربية مع عناوين فرعية إنجليزية مختصرة، واذكر أن التوقعات تقديرية.\n"
        "الأقسام المطلوبة (Markdown):\n"
        "1) الملخص التنفيذي | Executive Summary\n"
        "2) توقع المبيعات والإيرادات | Sales & Revenue Forecast (أرقام، اتجاه، ثقة)\n"
        "3) توقع الطلب على المنتجات/الخدمات | Product Demand Forecast\n"
        "4) مخاطر انخفاض الأرباح | Profit Decline Risk (مستوى الخطر، إشارات، إجراءات وقائية)\n"
        "5) توقع سلوك العملاء | Customer Behavior Prediction (شرائح، churn risk، فرص)\n"
        "6) خطة عمل 30 يوم | 30-Day Action Plan (5 خطوات عملية بأولوية)\n"
        "7) افتراضات وحدود | Assumptions & Limits\n\n"
        "JSON:\n```json\n" + json.dumps(ctx, ensure_ascii=False, indent=2) + "\n```"
    )


def _call_subprocess(model: str, api_key: str, prompt: str, timeout: float) -> str:
    payload = json.dumps({
        "model": model, "key": api_key, "prompt": prompt,
        "system": PREDICTIVE_SYSTEM, "http_timeout": max(5, int(timeout) - 2),
    })
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT], input=payload,
        capture_output=True, text=True, encoding="utf-8", timeout=timeout,
        check=False,
    )
    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        raise RuntimeError((proc.stderr or "no response")[-180:])
    data = json.loads(lines[-1])
    if not data.get("ok"):
        code = data.get("code")
        if code:
            raise urllib.error.HTTPError("gemini", int(code), str(data.get("error")), None, None)
        raise RuntimeError(str(data.get("error")))
    return data["text"]


def generate_ai_insights(ctx: dict, api_key: str, model: str = DEFAULT_MODEL,
                         deadline_seconds: float = 35.0) -> tuple[str, dict]:
    prompt = _build_prompt(ctx)
    if not api_key:
        return fallback_insights(ctx), {"source": "local", "model": None, "note": "لا يوجد مفتاح للذكاء الاصطناعي."}
    started = time.monotonic()
    errors: list[str] = []
    for candidate in [model] + [m for m in MODELS if m != model]:
        remaining = deadline_seconds - (time.monotonic() - started)
        if remaining < 6:
            break
        try:
            text = _call_subprocess(candidate, api_key, prompt, timeout=min(28, remaining))
            return text, {"source": "ai", "model": candidate, "note": None}
        except urllib.error.HTTPError as e:
            errors.append(f"{candidate}:HTTP{e.code}")
            if e.code in (400, 401, 403):
                break
        except Exception as e:
            errors.append(f"{candidate}:{type(e).__name__}")
    return fallback_insights(ctx), {"source": "local", "model": None, "note": "؛ ".join(errors[:3]) or "fallback"}


def fallback_insights(ctx: dict) -> str:
    cur = ctx.get("currency", "JOD")
    h = ctx.get("horizon_days", 30)
    rev = ctx.get("revenue_forecast", {})
    prof = ctx.get("profit_forecast", {})
    risk = ctx.get("profit_risk", {})
    cust = ctx.get("customer_behavior", {})
    demand = ctx.get("top_demand", [])
    lines = [
        "## 1) الملخص التنفيذي | Executive Summary",
        f"- أفق التوقع: **{h} يومًا** (تقديري مبني على الأنماط التاريخية).",
        f"- الإيراد المتوقع: **{_money(rev.get('total'), cur)}** (متوسط يومي {_money(rev.get('avg_daily'), cur)}، نمو { _pct(rev.get('growth_pct')) }).",
        f"- الربح المتوقع: **{_money(prof.get('total'), cur)}** (نمو {_pct(prof.get('growth_pct'))}).",
        f"- مخاطر انخفاض الأرباح: **{risk.get('risk','—')}**.",
        "",
        "## 2) توقع المبيعات والإيرادات | Sales & Revenue Forecast",
        f"- إجمالي متوقع: {_money(rev.get('total'), cur)} خلال {h} يومًا.",
        "- يعتمد على الاتجاه الخطي + الموسمية الأسبوعية؛ راجع الفاصل في الرسم البياني.",
        "",
        "## 3) توقع الطلب على المنتجات/الخدمات | Product Demand Forecast",
    ]
    if demand:
        lines.extend(
            f"- **{d.get('service', '—')}**: طلب متوقع {d.get('projected_orders', '—')} "
            f"({d.get('trend', '—')})، إيراد متوقع {_money(d.get('projected_revenue'), cur)}."
            for d in demand[:5]
        )
    else:
        lines.append("- لا تتوفر بيانات كافية لتجزئة الطلب حسب المنتج/الخدمة.")
    lines += [
        "",
        "## 4) مخاطر انخفاض الأرباح | Profit Decline Risk",
        f"- المستوى: **{risk.get('risk','—')}** | ميل يومي: {risk.get('slope_daily','—')} | هامش حديث: {_pct(risk.get('recent_margin_pct'))}.",
    ]
    for s in (risk.get("signals") or [])[:5]:
        lines.append(f"- {s}")
    if risk.get("days_to_negative"):
        lines.append(f"- ⚠️ تقدير اقتراب من منطقة حرجة خلال ~{risk['days_to_negative']} يومًا إذا استمر الاتجاه.")
    lines += [
        "- إجراءات وقائية: مراقبة المصاريف المتغيرة أسبوعيًا، تحسين التسعير/المزيج، وتتبع الهامش اليومي.",
        "",
        "## 5) توقع سلوك العملاء | Customer Behavior Prediction",
    ]
    if cust.get("available"):
        sm = cust.get("summary", {})
        lines.append(f"- عملاء: {sm.get('total_customers','—')} | معرّضون للفقدان: {sm.get('at_risk_customers','—')} ({_pct(sm.get('at_risk_pct'))}).")
        for s in (cust.get("signals") or [])[:4]:
            lines.append(f"- {s}")
    else:
        lines.append("- لا تتوفر بيانات عملاء/تواريخ كافية لتحليل RFM وسلوك الشراء.")
    lines += [
        "",
        "## 6) خطة عمل 30 يوم | 30-Day Action Plan",
        "1. **مراقبة يومية** للإيراد/الربح مقابل خط التوقع وتنبيه عند انحراف > 15%.",
        "2. **تعزيز الطلب** على الخدمات الأعلى نموًا بعروض موجّهة.",
        "3. **ضبط التكاليف** في البنود الأعلى أثرًا عند ظهور إشارات هبوط الربح.",
        "4. **استعادة العملاء المعرّضين للفقدان** بحملة تواصل/ولاء مستهدفة.",
        "5. **مراجعة أسبوعية** للنموذج مع بيانات جديدة لتحسين دقة التوقع.",
        "",
        "## 7) افتراضات وحدود | Assumptions & Limits",
        "- التوقعات إحصائية (اتجاه+موسمية) ولا تلتقط حملات مفاجئة أو تغيرات سوقية كبيرة.",
        "- دقة سلوك العملاء تعتمد على جودة أعمدة العميل/التاريخ/القيمة.",
    ]
    return "\n".join(lines)
