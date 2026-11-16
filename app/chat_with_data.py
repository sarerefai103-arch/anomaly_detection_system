"""
chat_with_data.py
-----------------
مساعد محادثة يجيب عن أسئلة حول بيانات الأعمال بالأرقام.

- يبني سياقًا مضغوطًا من مخرجات `business_analytics.analyze()`
  (مؤشرات، اتجاه شهري، أعلى العملاء/الخدمات، أيام الانحراف، التوقّع).
- يجيب بالذكاء الاصطناعي عند توفّر مفتاح، وبديل محلي بالكلمات المفتاحية.
- لا يخترع أرقامًا: كل الأرقام تأتي من البيانات المحلّلة.

المطوّر: sari refai
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any

import ai_report as ai

SUGGESTIONS = [
    "ما إجمالي الإيراد والربح؟",
    "كيف كانت المبيعات شهريًا؟",
    "من أهم العملاء؟",
    "ما أعلى الخدمات إيرادًا؟",
    "هل توجد أيام فيها انحراف؟",
    "ما توقّع الإيراد للفترة القادمة؟",
]


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def money(value: Any, currency: str = "") -> str:
    number = _num(value)
    if number is None:
        return "—"
    text = f"{number:,.2f}"
    return f"{text} {currency}".strip()


def pct(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "—"
    return f"{number:.1f}%"


# ============================================================
# 1) السياق المضغوط
# ============================================================
def build_context(result: dict, market: dict) -> dict:
    """يبني سياقًا صغيرًا يكفي للإجابة عن معظم الأسئلة."""
    kpis = result.get("kpis", {}) or {}
    score = result.get("score", {}) or {}
    daily = result.get("daily")
    monthly = result.get("monthly")

    monthly_rows: list[dict] = []
    if monthly is not None and not monthly.empty:
        for _, row in monthly.tail(12).iterrows():
            monthly_rows.append({
                "month": str(row.get("label", row.get("date", ""))),
                "revenue": round(float(row.get("revenue", 0) or 0), 2),
                "profit": round(float(row.get("profit", 0) or 0), 2),
                "margin_pct": round(float(row.get("margin", 0) or 0) * 100, 1),
                "orders": int(row.get("orders", 0) or 0),
            })

    clients: list[dict] = []
    services: list[dict] = []
    for source, sink in ((result.get("clients"), clients), (result.get("services"), services)):
        if source is None or source.empty:
            continue
        for _, row in source.head(5).iterrows():
            sink.append({
                "name": str(row.get("name", "")),
                "revenue": round(float(row.get("revenue", 0) or 0), 2),
                "orders": int(row.get("orders", 0) or 0),
                "share_pct": round(float(row.get("share", 0) or 0) * 100, 1),
            })

    anomalies: list[dict] = []
    if daily is not None and not daily.empty and "is_anomaly" in daily.columns:
        flagged = daily[daily["is_anomaly"]]
        for _, row in flagged.tail(8).iterrows():
            anomalies.append({
                "date": str(row.get("date", ""))[:10],
                "revenue": round(float(row.get("revenue", 0) or 0), 2),
                "z_score": round(float(row.get("z_score", 0) or 0), 2),
            })

    forecast: dict = {}
    table = result.get("forecast")
    if table is not None and not table.empty:
        forecast = {
            "days": len(table),
            "expected_revenue": round(float(table["forecast"].sum()), 2),
            "expected_daily_avg": round(float(table["forecast"].mean()), 2),
            "start": str(table["date"].iloc[0])[:10],
            "end": str(table["date"].iloc[-1])[:10],
        }

    revenue = _num(kpis.get("revenue")) or 0.0
    expense = _num(kpis.get("expense")) or 0.0
    profit = _num(kpis.get("profit")) or 0.0

    return {
        "currency": market.get("currency", kpis.get("currency", "JOD")),
        "vat_rate_pct": round(float(market.get("vat_rate", 0) or 0) * 100, 2),
        "target_margin_pct": round(float(market.get("target_margin", 0) or 0) * 100, 1),
        "totals": {
            "revenue": round(revenue, 2),
            "expense": round(expense, 2),
            "profit": round(profit, 2),
            "margin_pct": round((profit / revenue * 100) if revenue else 0.0, 1),
            "orders": int(_num(kpis.get("orders")) or 0),
            "avg_ticket": round(_num(kpis.get("avg_ticket")) or 0.0, 2),
            "collection_rate_pct": round((_num(kpis.get("collection_rate")) or 0.0) * 100, 1),
            "growth_pct": None if kpis.get("growth") is None else round(float(kpis["growth"]) * 100, 1),
            "expense_ratio_pct": round((_num(kpis.get("expense_ratio")) or 0.0) * 100, 1),
            "vat_amount": round(_num(kpis.get("vat_amount")) or 0.0, 2),
        },
        "health": {
            "score": score.get("score"),
            "label": score.get("label"),
        },
        "top_clients": clients,
        "top_services": services,
        "top_client_share_pct": round((_num(kpis.get("top_client_share")) or 0.0) * 100, 1),
        "top_service_share_pct": round((_num(kpis.get("top_service_share")) or 0.0) * 100, 1),
        "monthly": monthly_rows,
        "anomaly_days": anomalies,
        "forecast": forecast,
        "break_even": {
            "daily": round(_num((result.get("break_even") or {}).get("daily")) or 0.0, 2),
            "monthly": round(_num((result.get("break_even") or {}).get("monthly")) or 0.0, 2),
        },
        "has_expense": bool(result.get("has_expense")),
        "has_date": bool(result.get("has_date")),
    }


def signature(ctx: dict) -> str:
    payload = json.dumps(ctx, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _context_block(ctx: dict) -> str:
    return json.dumps(ctx, ensure_ascii=False, indent=2)


# ============================================================
# 2) بناء السؤال
# ============================================================
_PROMPT = """أنت مساعد تحليل بيانات ذكي لأعمال تجارية.
لديك ملخص بيانات الأعمال التالي (JSON):

{context}

سؤال المستخدم:
{question}

القواعد:
- أجب بنفس لغة السؤال (عربي أو إنجليزي)، باختصار ووضوح.
- استخدم الأرقام الموجودة في الملخص فقط، ولا تخترع أي رقم.
- اذكر العملة عند ذكر مبالغ مالية.
- استخدم نقاطًا أو جدول Markdown قصير عند الحاجة.
- إذا كانت البيانات لا تكفي للإجابة، قل ذلك بوضوح واقترح ما يمكن حسابه.
"""


def build_prompt(ctx: dict, question: str) -> str:
    return _PROMPT.format(context=_context_block(ctx), question=question.strip())


# ============================================================
# 3) البديل المحلي (بدون مفتاح)
# ============================================================
def _has(question: str, words) -> bool:
    text = question.lower()
    return any(word in text for word in words)


def fallback_answer(ctx: dict, question: str) -> str:
    """إجابة محلية مبنية على الكلمات المفتاحية والأرقام المتاحة."""
    cur = ctx.get("currency", "")
    totals = ctx.get("totals", {})
    lines: list[str] = []

    if _has(question, ["ربح", "profit", "هامش", "margin"]):
        lines.append(
            f"**الربح | Profit:** {money(totals.get('profit'), cur)} "
            f"· هامش {pct(totals.get('margin_pct'))}"
        )
        if ctx.get("break_even", {}).get("monthly"):
            lines.append(
                f"- نقطة التعادل الشهرية | Monthly break-even: "
                f"{money(ctx['break_even']['monthly'], cur)}"
            )
        if totals.get("expense_ratio_pct") is not None:
            lines.append(
                f"- نسبة المصاريف من الإيراد | Expense ratio: "
                f"{pct(totals.get('expense_ratio_pct'))}"
            )

    if _has(question, ["إيراد", "ايراد", "مبيعات", "revenue", "sales", "دخل"]):
        lines.append(
            f"**الإيراد | Revenue:** {money(totals.get('revenue'), cur)} "
            f"من {totals.get('orders', 0):,} فاتورة"
        )
        if totals.get("growth_pct") is not None:
            lines.append(f"- نمو الإيراد | Revenue growth: {pct(totals.get('growth_pct'))}")
        lines.append(
            f"- متوسط الفاتورة | Average ticket: {money(totals.get('avg_ticket'), cur)}"
        )

    if _has(question, ["مصروف", "مصاريف", "تكلفة", "expense", "cost"]):
        lines.append(
            f"**المصاريف | Expenses:** {money(totals.get('expense'), cur)} "
            f"({pct(totals.get('expense_ratio_pct'))} من الإيراد)"
        )

    if _has(question, ["ضريبة", "vat", "tax"]):
        lines.append(
            f"**ضريبة القيمة المضافة | VAT ({pct(ctx.get('vat_rate_pct'))}):** "
            f"{money(totals.get('vat_amount'), cur)}"
        )

    if _has(question, ["عميل", "عملاء", "client", "customer"]):
        clients = ctx.get("top_clients", [])
        if clients:
            lines.append("**أهم العملاء | Top clients:**")
            lines.extend(
                f"- {item['name']}: {money(item['revenue'], cur)} "
                f"({pct(item['share_pct'])})"
                for item in clients[:5]
            )
            lines.append(
                f"- حصة أكبر عميل | Top client share: "
                f"{pct(ctx.get('top_client_share_pct'))}"
            )
        else:
            lines.append("لا يوجد عمود عملاء في البيانات. | No customer column found.")

    if _has(question, ["خدمة", "خدمات", "منتج", "products", "service", "category"]):
        services = ctx.get("top_services", [])
        if services:
            lines.append("**أعلى الخدمات/المنتجات | Top services:**")
            lines.extend(
                f"- {item['name']}: {money(item['revenue'], cur)} "
                f"({pct(item['share_pct'])})"
                for item in services[:5]
            )
        else:
            lines.append("لا يوجد عمود خدمات/منتجات. | No product/service column found.")

    if _has(question, ["شهر", "شهري", "month", "monthly", "أفضل", "best"]):
        rows = ctx.get("monthly", [])
        if rows:
            best = max(rows, key=lambda r: r["revenue"])
            worst = min(rows, key=lambda r: r["revenue"])
            lines.append("**الاتجاه الشهري | Monthly trend:**")
            lines.append(
                f"- أفضل شهر | Best month: {best['month']} — "
                f"{money(best['revenue'], cur)} (ربح {money(best['profit'], cur)})"
            )
            lines.append(
                f"- أضعف شهر | Weakest month: {worst['month']} — "
                f"{money(worst['revenue'], cur)} (ربح {money(worst['profit'], cur)})"
            )
            lines.extend(
                f"- {row['month']}: إيراد {money(row['revenue'], cur)} · "
                f"ربح {money(row['profit'], cur)} · هامش {pct(row['margin_pct'])}"
                for row in rows[-6:]
            )

    if _has(question, ["انحراف", "شاذ", "anomaly", "outlier", "غير طبيعي"]):
        anomalies = ctx.get("anomaly_days", [])
        if anomalies:
            lines.append(f"**أيام فيها انحراف | Anomalous days ({len(anomalies)}):**")
            lines.extend(
                f"- {item['date']}: إيراد {money(item['revenue'], cur)} "
                f"(z={item['z_score']})"
                for item in anomalies[-5:]
            )
        else:
            lines.append("لم تُكتشف أيام انحراف واضحة. | No clear revenue anomalies found.")

    if _has(question, ["توقع", "توقّع", "forecast", "predict", "متوقع", "قادم"]):
        forecast = ctx.get("forecast") or {}
        if forecast:
            lines.append(
                f"**التوقّع | Forecast ({forecast['days']} يوم):** "
                f"{money(forecast['expected_revenue'], cur)} إجمالًا، "
                f"بمتوسط يومي {money(forecast['expected_daily_avg'], cur)} "
                f"({forecast['start']} → {forecast['end']})"
            )
        else:
            lines.append("لا توجد بيانات تاريخ كافية للتوقّع. | Not enough dated data to forecast.")

    if _has(question, ["صحة", "تقييم", "health", "score"]):
        health = ctx.get("health", {})
        lines.append(
            f"**تقييم صحة الأعمال | Business health:** "
            f"{health.get('score', '—')} / 100 — {health.get('label', '—')}"
        )

    if not lines:
        # ملخص عام + اقتراحات
        lines.append(
            f"**ملخص سريع | Quick summary:** إيراد {money(totals.get('revenue'), cur)} · "
            f"مصاريف {money(totals.get('expense'), cur)} · "
            f"ربح {money(totals.get('profit'), cur)} · هامش {pct(totals.get('margin_pct'))}"
        )
        services = ctx.get("top_services", [])
        if services:
            lines.append(
                "أعلى خدمة | Top service: "
                f"{services[0]['name']} ({pct(services[0]['share_pct'])})"
            )
        lines.append(
            "\nجرّب أسئلة مثل: " + " · ".join(f"«{s}»" for s in SUGGESTIONS[:3])
        )

    lines.append("\n> ℹ️ إجابة محلية مبنية على أرقام بياناتك (بدون اتصال).")
    return "\n".join(lines)


# ============================================================
# 4) الإجابة بالذكاء الاصطناعي مع بديل
# ============================================================
def answer(
    question: str,
    ctx: dict,
    api_key: str = "",
    model: str | None = None,
    deadline_seconds: float = 40.0,
) -> tuple[str, dict]:
    """يعيد (الإجابة، معلومات المصدر)."""
    question = (question or "").strip()
    if not question:
        return "", {"source": "local", "model": None, "note": "سؤال فارغ."}

    model = model or ai.DEFAULT_MODEL
    if not api_key:
        return fallback_answer(ctx, question), {
            "source": "local", "model": None,
            "note": "لا يوجد مفتاح للذكاء الاصطناعي.",
        }

    prompt = build_prompt(ctx, question)
    started = time.monotonic()
    errors: list[str] = []
    for candidate in [model] + [m for m in ai.MODELS if m != model]:
        remaining = deadline_seconds - (time.monotonic() - started)
        if remaining < 6:
            errors.append("انتهت الميزانية الزمنية")
            break
        try:
            text = ai.call_in_subprocess(candidate, api_key, prompt, timeout=min(28, remaining))
            return text, {"source": "ai", "model": candidate, "note": None}
        except Exception as err:
            errors.append(f"{type(err).__name__}: {err}")

    return fallback_answer(ctx, question), {
        "source": "local", "model": None,
        "note": "تعذّر الوصول إلى خدمة الذكاء الاصطناعي. " + "؛ ".join(errors[:2]),
    }
