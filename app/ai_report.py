"""
ai_report.py
------------
توليد التقرير المالي الذكي بالذكاء الاصطناعي.

- يقرأ المفتاح من `.streamlit/secrets.toml` أو متغيرات البيئة (`AI_API_KEY` / `GEMINI_API_KEY` للتوافق).
- يستخدم REST مباشرة (urllib) بدون أي مكتبة إضافية.
- يجرّب عدة مسارات تلقائيًا عند ازدحام الخدمة (503).
- إذا تعذّر الاتصال، يبني تقريرًا محليًا احتياطيًا من نفس الأرقام.

المطوّر: sari refai
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)

# المسارات المجرّبة بالترتيب (داخلي). تُعرض تسميات عامة في الواجهة.
MODELS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]
DEFAULT_MODEL = MODELS[0]

MODEL_LABELS = {
    "gemini-3.5-flash": "النموذج الذكي السريع | Fast AI Model",
    "gemini-3.6-flash": "النموذج الذكي المتوازن | Balanced AI Model",
    "gemini-3.7-flash": "النموذج الذكي المتقدم | Advanced AI Model",
    "gemini-3.8-flash": "النموذج الذكي الأحدث | Latest AI Model",
    "gemini-flash-latest": "النموذج الذكي التلقائي | Auto AI Model",
}


def model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, "نموذج ذكاء اصطناعي | AI Model")

SYSTEM_PROMPT = (
    "أنت محلل مالي خبير في تحليل أعمال السوق المحلي (الأردن والخليج) والعالمي. "
    "تكتب تقارير مالية واضحة، مهنية، وقابلة للتنفيذ، وتعتمد حصريًا على الأرقام المعطاة. "
    "لا تخترع أي رقم أو عميل أو خدمة غير موجودة في البيانات."
)


# ============================================================
# المفتاح
# ============================================================
def resolve_api_key(session_key: str | None = None) -> str:
    """يرجع المفتاح من: إدخال المستخدم ← secrets ← متغيرات البيئة (مع توافق)."""
    if session_key and session_key.strip():
        return session_key.strip()

    try:
        import streamlit as st

        for name in ("AI_API_KEY", "GEMINI_API_KEY"):
            if name in st.secrets:
                return str(st.secrets[name]).strip()
    except Exception:
        pass

    return os.environ.get("AI_API_KEY", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()


# ============================================================
# بناء سياق التحليل
# ============================================================
def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else round(number, 3)  # استبعاد NaN


def build_context(result: dict, market: dict) -> dict:
    """يحوّل نتائج التحليل إلى سياق صغير ومنظّم يُرسل إلى Gemini."""
    kpis = result["kpis"]
    daily = result["daily"]
    monthly = result["monthly"]

    return {
        "developer": "sari refai",
        "currency": market["currency"],
        "vat_rate": market["vat_rate"],
        "target_margin": market["target_margin"],
        "forecast_days": market["forecast_days"],
        "totals": {
            "revenue": _num(kpis["revenue"]),
            "expense": _num(kpis["expense"]),
            "profit": _num(kpis["profit"]),
            "margin_pct": _num(kpis["margin"] * 100),
            "orders": int(kpis["orders"]),
            "avg_ticket": _num(kpis["avg_ticket"]),
            "collection_rate_pct": _num(kpis["collection_rate"] * 100),
            "growth_pct": None if kpis["growth"] is None else _num(kpis["growth"] * 100),
            "growth_period_days": int(kpis["growth_period_days"]),
            "vat_amount": _num(kpis["vat_amount"]),
            "revenue_ex_vat": _num(kpis["revenue_ex_vat"]),
            "expense_ratio_pct": _num(kpis["expense_ratio"] * 100),
            "pending_value": _num(kpis["pending_value"]),
            "lost_value": _num(kpis["lost_value"]),
        },
        "market_health": {
            "score": result["score"]["score"],
            "label": result["score"]["label"],
        },
        "break_even": {
            "daily": _num(result["break_even"].get("daily", 0)),
            "monthly": _num(result["break_even"].get("monthly", 0)),
        },
        "concentration": {
            "top_client_share_pct": _num(kpis.get("top_client_share", 0) * 100),
            "top_service_share_pct": _num(kpis.get("top_service_share", 0) * 100),
        },
        "monthly": [
            {
                "month": str(row["label"]),
                "revenue": _num(row["revenue"]),
                "expense": _num(row["expense"]),
                "profit": _num(row["profit"]),
                "margin_pct": _num(row["margin"] * 100),
                "orders": int(row["orders"]),
            }
            for _, row in monthly.iterrows()
        ],
        "top_services": [
            {
                "name": str(row["name"]),
                "revenue": _num(row["revenue"]),
                "share_pct": _num(row["share"] * 100),
            }
            for _, row in result["services"].head(6).iterrows()
        ],
        "top_clients": [
            {
                "name": str(row["name"]),
                "revenue": _num(row["revenue"]),
                "share_pct": _num(row["share"] * 100),
            }
            for _, row in result["clients"].head(6).iterrows()
        ],
        "status": [
            {
                "status": str(row["label"]),
                "value": _num(row["value"]),
                "invoices": int(row["orders"]),
            }
            for _, row in result["status"].iterrows()
        ],
        "anomalies": [
            {
                "date": str(row["date"])[:10],
                "revenue": _num(row["revenue"]),
                "z_score": _num(row["z_score"]),
            }
            for _, row in daily[daily["is_anomaly"]].sort_values("z_score", ascending=False).head(8).iterrows()
        ] if not daily.empty else [],
        "forecast": {
            "days": len(result["forecast"]),
            "avg_daily": _num(result["forecast"]["forecast"].mean()) if not result["forecast"].empty else None,
            "total": _num(result["forecast"]["forecast"].sum()) if not result["forecast"].empty else None,
        },
        "data": {
            "rows": len(result["prepared"]),
            "has_expense": result["has_expense"],
            "has_date": result["has_date"],
            "first_day": str(daily["date"].min())[:10] if not daily.empty else None,
            "last_day": str(daily["date"].max())[:10] if not daily.empty else None,
        },
    }


def signature(context: dict) -> str:
    """بصمة ثابتة للبيانات حتى لا يُعاد توليد التقرير عند كل تحديث."""
    payload = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


# ============================================================
# صياغة الطلب
# ============================================================
def build_prompt(context: dict) -> str:
    data = json.dumps(context, ensure_ascii=False, indent=2)
    return f"""حلّل بيانات الأعمال التالية وأكتب تقريرًا ماليًا ذكيًا.

[بيانات الأعمال — JSON]
{data}

[المطلوب]
اكتب تقريرًا بالعربية الفصحى المبسطة، بصيغة Markdown، وبهذه الأقسام بالترتيب:

## 1. الملخص التنفيذي | Executive Summary
3–4 أسطر: الإيراد، الربح، الهامش، وأهم رسالة للقرار.

## 2. قراءة المؤشرات | KPI Reading
فسّر: الإيراد، المصاريف، صافي الربح، هامش الربح، متوسط الفاتورة، نسبة التحصيل، النمو، ضريبة المبيعات، ونقطة التعادل. اذكر كل رقم مع وحدته.

## 3. الاتجاهات والموسمية | Trends
حلّل الأداء الشهري، أفضل وأضعف شهر، وأثر يوم الأسبوع إن توفر. استخدم جدول Markdown للأشهر.

## 4. المخاطر | Risks
تحدّث عن: تركّز العملاء، تركّز الخدمات، الفواتير المعلّقة، القيمة المفقودة (ملغاة/مرتجعة)، والأيام الشاذة في الإيراد.

## 5. التوقّع | Forecast
لخّص توقّع الإيراد للأيام القادمة (المتوسط اليومي والإجمالي) ونبّه أن التوقّع تقديري.

## 6. التوصيات | Recommendations
5 توصيات عملية مرقّمة، كل واحدة سطر أو سطران، مرتبطة بأرقام البيانات ومفيدة للسوق المحلي والعالمي.

[قيود]
- استخدم الأرقام الموجودة في JSON فقط، ولا تخترع أرقامًا.
- totals.revenue شامل ضريبة المبيعات. totals.revenue_ex_vat هو الإيراد بعد استبعاد الضريبة، وtotals.vat_amount هو مبلغ الضريبة. لا تخلط بينها ولا تجعل الضريبة صفرًا إن كانت أكبر من صفر.
- totals.expense هي التكلفة الإجمالية، وtotals.profit = revenue - expense.
- اذكر العملة {context["currency"]} بعد الأرقام المالية.
- لا تزد عن 700 كلمة.
- لا تكتب أي مقدمات أو خاتمة خارج الأقسام.
- اختم التقرير بسطر: «أُعدّ بواسطة sari refai — نظام تحليل البيانات الذكي».
"""


# ============================================================
# استدعاء Gemini
# ============================================================
def _extract_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        reason = (payload.get("promptFeedback") or {}).get("blockReason", "EMPTY")
        raise RuntimeError(f"لم يُرجِع النموذج أي نص ({reason}).")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "\n".join(part.get("text", "") for part in parts).strip()
    if not text:
        reason = candidates[0].get("finishReason", "UNKNOWN")
        raise RuntimeError(f"إجابة فارغة من النموذج ({reason}).")
    return text


def call_gemini(model: str, api_key: str, prompt: str, timeout: int = 90) -> str:
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.35,
            "topP": 0.95,
            "maxOutputTokens": 4096,
        },
    }
    request = urllib.request.Request(  # noqa: S310 — عنوان ثابت https
        ENDPOINT.format(model=model, key=api_key),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return _extract_text(json.load(response))


_WORKER_SCRIPT = r"""
import json, sys, urllib.error, urllib.request

cfg = json.load(sys.stdin)
url = ("https://generativelanguage.googleapis.com/v1beta/models/"
       + cfg["model"] + ":generateContent?key=" + cfg["key"])
body = json.dumps({
    "systemInstruction": {"parts": [{"text": cfg["system"]}]},
    "contents": [{"role": "user", "parts": [{"text": cfg["prompt"]}]}],
    "generationConfig": {
        "temperature": 0.35,
        "topP": 0.95,
        "maxOutputTokens": 4096,
    },
}).encode("utf-8")
request = urllib.request.Request(
    url, data=body, headers={"Content-Type": "application/json"}, method="POST"
)
try:
    with urllib.request.urlopen(request, timeout=cfg.get("http_timeout", 30)) as response:
        payload = json.load(response)
    candidates = payload.get("candidates") or []
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    text = "\n".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise RuntimeError("empty response")
    print(json.dumps({"ok": True, "text": text}))
except urllib.error.HTTPError as err:
    print(json.dumps({"ok": False, "code": err.code, "error": "HTTP %s" % err.code}))
except Exception as err:  # noqa: BLE001
    print(json.dumps({"ok": False, "code": None,
                      "error": type(err).__name__ + ": " + str(err)[:200]}))
"""


def call_in_subprocess(model: str, api_key: str, prompt: str, timeout: float) -> str:
    """ينفّذ الطلب في عملية منفصلة مع قتل صارم عند انتهاء المهلة.

    استخدام عملية منفصلة يضمن ألا يعلّق الخادم واجهة Streamlit حتى لو
    لم يستجب الاتصال (SSL قد يحتجز الـ GIL داخل نفس العملية).
    """
    payload = json.dumps({
        "model": model,
        "key": api_key,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "http_timeout": max(5, int(timeout) - 2),
    })
    process = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )
    lines = (process.stdout or "").strip().splitlines()
    if not lines:
        raise RuntimeError((process.stderr or "لا يوجد رد من العملية")[-200:])

    data = json.loads(lines[-1])
    if not data.get("ok"):
        code = data.get("code")
        if code:
            raise urllib.error.HTTPError(
                "gemini", int(code), str(data.get("error")), None, None
            )
        raise RuntimeError(str(data.get("error")))
    return data["text"]


def generate_report(
    context: dict,
    api_key: str,
    model: str = DEFAULT_MODEL,
    allow_fallback: bool = True,
    deadline_seconds: float = 35.0,
) -> tuple[str, dict]:
    """يولّد التقرير. يعيد (النص، معلومات المصدر).

    يجرّب النموذج المختار ثم بقية النماذج عند 503/429/404 فقط،
    ضمن ميزانية زمنية محددة، وإن فشل الجميع يستخدم التقرير المحلي.
    """
    prompt = build_prompt(context)

    if not api_key:
        report = fallback_report(context)
        return report, {"source": "local", "model": None,
                        "note": "لا يوجد مفتاح للذكاء الاصطناعي، تم استخدام التقرير المحلي."}

    started = time.monotonic()
    order = [model] + [m for m in MODELS if m != model]
    errors: list[str] = []

    for candidate in order:
        remaining = deadline_seconds - (time.monotonic() - started)
        if remaining < 6:
            errors.append("انتهت الميزانية الزمنية")
            break

        try:
            text = call_in_subprocess(
                candidate, api_key, prompt, timeout=min(28, remaining)
            )
            return text, {"source": "ai", "model": candidate, "note": None}
        except urllib.error.HTTPError as err:
            errors.append(f"{candidate}: HTTP {err.code}")
            # مفتاح أو طلب غير صالح — لا فائدة من تجربة الباقي
            if err.code in (400, 401, 403):
                break
            # 404/503/429: جرّب النموذج التالي بسرعة
            continue
        except subprocess.TimeoutExpired:
            errors.append(f"{candidate}: انتهت المهلة")
            # لا نُطيل الانتظار؛ التقرير المحلي جاهز فورًا
            break
        except Exception as err:
            errors.append(f"{candidate}: {type(err).__name__}")
            break

    if not allow_fallback:
        raise RuntimeError("؛ ".join(errors) or "فشل الاتصال بخدمة الذكاء الاصطناعي.")

    report = fallback_report(context)
    return report, {
        "source": "local",
        "model": None,
        "note": "تعذّر الوصول إلى خدمة الذكاء الاصطناعي، تم استخدام التقرير المحلي. " + "؛ ".join(errors[:3]),
    }


# ============================================================
# التقرير المحلي الاحتياطي
# ============================================================
def _money(value, currency: str) -> str:
    if value is None:
        return "—"
    return f"{value:,.0f} {currency}"


def _pct(value) -> str:
    return "—" if value is None else f"{value:.1f}%"


def fallback_report(context: dict) -> str:
    """تقرير مالي مبني على قواعد — يعمل دون إنترنت أو مفتاح."""
    totals = context["totals"]
    cur = context["currency"]
    months = context["monthly"]
    services = context["top_services"]
    clients = context["top_clients"]
    anomalies = context["anomalies"]
    forecast = context["forecast"]
    concentration = context["concentration"]
    score = context["market_health"]

    best_month = max(months, key=lambda m: m["revenue"]) if months else None
    worst_month = min(months, key=lambda m: m["revenue"]) if months else None

    lines: list[str] = []
    lines.append("## 1. الملخص التنفيذي | Executive Summary")
    lines.append(
        f"حقّق النشاط إيرادًا إجماليًا قدره **{_money(totals['revenue'], cur)}** "
        f"من **{totals['orders']:,}** فاتورة، بصافي ربح **{_money(totals['profit'], cur)}** "
        f"وهامش ربح **{_pct(totals['margin_pct'])}**."
    )
    if totals["growth_pct"] is not None:
        direction = "نموًا" if totals["growth_pct"] >= 0 else "تراجعًا"
        lines.append(
            f"سجّل الإيراد {direction} بنسبة **{totals['growth_pct']:+.1f}%** "
            f"مقارنة بالفترة السابقة الممتدة {totals['growth_period_days']} يومًا."
        )
    lines.append(
        f"تقييم صحة الأعمال **{score['score']}/100** ({score['label']})."
    )

    lines.append("\n## 2. قراءة المؤشرات | KPI Reading")
    lines.append(f"- الإيراد: **{_money(totals['revenue'], cur)}**")
    lines.append(f"- المصاريف: **{_money(totals['expense'], cur)}** "
                 f"({_pct(totals['expense_ratio_pct'])} من الإيراد)")
    lines.append(f"- صافي الربح: **{_money(totals['profit'], cur)}**")
    lines.append(f"- هامش الربح: **{_pct(totals['margin_pct'])}** "
                 f"(الهدف {_pct(context['target_margin'] * 100)})")
    lines.append(f"- متوسط الفاتورة: **{_money(totals['avg_ticket'], cur)}**")
    lines.append(f"- نسبة التحصيل: **{_pct(totals['collection_rate_pct'])}**")
    lines.append(f"- ضريبة المبيعات {_pct(context['vat_rate'] * 100)}: "
                 f"**{_money(totals['vat_amount'], cur)}**")
    lines.append(f"- الإيراد بدون ضريبة: **{_money(totals['revenue_ex_vat'], cur)}**")
    lines.append(
        f"- نقطة التعادل: **{_money(context['break_even']['daily'], cur)}** يوميًا "
        f"و**{_money(context['break_even']['monthly'], cur)}** شهريًا"
    )
    if not context["data"]["has_expense"]:
        lines.append("- تنبيه: لا يوجد عمود تكلفة في الملف، لذا الربح محسوب مساويًا للإيراد.")

    lines.append("\n## 3. الاتجاهات والموسمية | Trends")
    if months:
        lines.append("| الشهر | الإيراد | المصاريف | الربح | الهامش |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.extend(
            (
                f"| {month['month']} | {_money(month['revenue'], cur)} | "
                f"{_money(month['expense'], cur)} | {_money(month['profit'], cur)} | "
                f"{_pct(month['margin_pct'])} |"
            )
            for month in months
        )
        lines.append(
            f"\nأفضل شهر كان **{best_month['month']}** بإيراد "
            f"{_money(best_month['revenue'], cur)}، وأضعف شهر "
            f"**{worst_month['month']}** بإيراد {_money(worst_month['revenue'], cur)}."
        )
    else:
        lines.append("لا يتوفر عمود تاريخ لتحليل الاتجاهات الزمنية.")

    lines.append("\n## 4. المخاطر | Risks")
    lines.append(
        f"- تركّز العملاء: أكبر عميل يمثّل **{_pct(concentration['top_client_share_pct'])}** "
        "من الإيراد."
    )
    lines.append(
        f"- تركّز الخدمات: أكبر خدمة تمثّل **{_pct(concentration['top_service_share_pct'])}** "
        "من الإيراد."
    )
    if totals["pending_value"]:
        lines.append(f"- فواتير معلّقة بقيمة **{_money(totals['pending_value'], cur)}** تحتاج متابعة تحصيل.")
    if totals["lost_value"]:
        lines.append(f"- قيمة ملغاة/مرتجعة **{_money(totals['lost_value'], cur)}**.")
    if anomalies:
        lines.append(
            f"- رُصدت **{len(anomalies)}** أيام ذات إيراد شاذ، أعلاها "
            f"{anomalies[0]['date']} بقيمة {_money(anomalies[0]['revenue'], cur)}."
        )
    if not totals["pending_value"] and not totals["lost_value"] and not anomalies:
        lines.append("- لا مؤشرات خطر جوهرية في الفواتير أو الأيام الشاذة.")

    lines.append("\n## 5. التوقّع | Forecast")
    if forecast.get("avg_daily") is not None:
        lines.append(
            f"يتوقّع النموذج متوسط إيراد يومي **{_money(forecast['avg_daily'], cur)}** "
            f"للأيام {forecast['days']} القادمة، بإجمالي تقديري "
            f"**{_money(forecast['total'], cur)}**. التوقّع تقديري ويتأثر بالموسمية."
        )
    else:
        lines.append("التوقّع غير متاح لعدم وجود عمود تاريخ كافٍ.")

    lines.append("\n## 6. التوصيات | Recommendations")
    recommendations: list[str] = []
    if totals["margin_pct"] is not None and totals["margin_pct"] < context["target_margin"] * 100:
        recommendations.append(
            "رفع الهامش بإعادة تسعير الخدمات منخفضة الربح أو التفاوض على تكاليف الموردين."
        )
    else:
        recommendations.append(
            "الحفاظ على الهامش الحالي مع إعادة استثمار جزء من الربح في الخدمات الأعلى قيمة."
        )
    if concentration["top_client_share_pct"] and concentration["top_client_share_pct"] > 35:
        recommendations.append(
            "تنويع قاعدة العملاء لتقليل مخاطر الاعتماد على عميل واحد."
        )
    if totals["collection_rate_pct"] is not None and totals["collection_rate_pct"] < 90:
        recommendations.append(
            "تفعيل متابعة آلية للفواتير المعلّقة وتحفيز الدفع المبكر."
        )
    if totals["growth_pct"] is not None and totals["growth_pct"] < 0:
        recommendations.append(
            "مراجعة أسباب تراجع الإيراد وزيادة الحملات في الأيام الأضعف."
        )
    if services:
        recommendations.append(
            f"التركيز على الخدمة الأعلى إيرادًا: {services[0]['name']} "
            f"بنسبة {_pct(services[0]['share_pct'])} من الإيراد كخدمة رئيسية."
        )
    if clients:
        recommendations.append(
            f"بناء برنامج ولاء لأفضل العملاء بدءًا من {clients[0]['name']} "
            f"بنسبة {_pct(clients[0]['share_pct'])} من الإيراد."
        )
    recommendations.append(
        "مراقبة الأيام الشاذة أسبوعيًا وربطها بالحملات أو المواسم لتحسين التخطيط."
    )
    for index, item in enumerate(recommendations[:5], start=1):
        lines.append(f"{index}. {item}")

    lines.append("\n---")
    lines.append("أُعدّ بواسطة sari refai — نظام تحليل البيانات الذكي")
    return "\n".join(lines)
