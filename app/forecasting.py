"""
forecasting.py
--------------
تقييم دقة التوقعات (Backtesting) + محاكي السيناريوهات (What-if).

- `backtest()`: يقارن عدة نماذج توقّع على بيانات لم يرها النموذج ويحسب
  MAE / RMSE / MAPE / Bias.
- `what_if()`: يحاكي أثر تغيير السعر/الكمية/المصاريف على الإيراد والربح.

المطوّر: sari refai
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

MODEL_LABELS = {
    "naive": "آخر قيمة | Naive",
    "seasonal_naive": "موسمي أسبوعي | Seasonal Naive",
    "moving_average": "متوسط متحرك | Moving Average",
    "linear_seasonal": "اتجاه + موسمية | Linear + Seasonality",
}


def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(number) or math.isinf(number)) else round(number, 3)


def _series(daily: pd.DataFrame, column: str) -> pd.Series:
    if daily.empty or column not in daily.columns:
        return pd.Series(dtype=float)
    series = (
        daily.set_index("date")[column]
        .astype(float)
        .asfreq("D")
        .interpolate(limit_direction="both")
    )
    return series.dropna()


# ============================================================
# 1) نماذج التوقع
# ============================================================
def _naive(train: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(float(train.iloc[-1]), horizon)


def _seasonal_naive(train: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    if len(train) < season:
        return _naive(train, horizon)
    last_season = train.iloc[-season:].to_numpy(dtype=float)
    return np.array([last_season[i % season] for i in range(horizon)])


def _moving_average(train: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    window = min(window, len(train))
    return np.repeat(float(train.iloc[-window:].mean()), horizon)


def _linear_seasonal(train: pd.Series, horizon: int, column: str) -> np.ndarray:
    y = train.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    if len(y) < 4:
        return _naive(train, horizon)

    slope, intercept = np.polyfit(x, y, 1)
    trend = slope * x + intercept
    residual = y - trend
    weekday = train.index.dayofweek
    seasonal = pd.Series(residual, index=weekday).groupby(level=0).mean()

    future = pd.date_range(train.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    fx = np.arange(len(y), len(y) + horizon, dtype=float)
    forecast = slope * fx + intercept + np.array(
        [seasonal.get(day, 0.0) for day in future.dayofweek]
    )
    if column == "profit":
        return forecast
    return np.clip(forecast, 0, None)


MODELS = {
    "naive": _naive,
    "seasonal_naive": _seasonal_naive,
    "moving_average": _moving_average,
    "linear_seasonal": _linear_seasonal,
}


# ============================================================
# 2) Backtesting
# ============================================================
def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    mask = np.abs(actual) > 1e-6
    mape = float(np.mean(np.abs(errors[mask] / actual[mask])) * 100) if mask.any() else None

    bias = float(np.mean(errors))
    return {
        "mae": _num(mae),
        "rmse": _num(rmse),
        "mape": _num(mape),
        "bias": _num(bias),
    }


def reliability(mape: float | None) -> tuple[str, str]:
    """يعيد (التسمية، المستوى) بناءً على MAPE."""
    if mape is None:
        return "غير محدد | N/A", "unknown"
    if mape <= 10:
        return "دقة عالية | High accuracy", "good"
    if mape <= 20:
        return "دقة جيدة | Good accuracy", "ok"
    if mape <= 35:
        return "دقة متوسطة | Fair accuracy", "warn"
    return "دقة منخفضة | Low accuracy", "bad"


def backtest(
    daily: pd.DataFrame,
    column: str = "revenue",
    test_days: int = 14,
    min_train_days: int = 21,
) -> dict[str, Any]:
    """يقيس دقة النماذج على آخر `test_days` لم يُستخدم في التدريب."""
    series = _series(daily, column)
    out: dict[str, Any] = {
        "available": False,
        "column": column,
        "test_days": int(test_days),
        "train_days": None,
        "metrics": pd.DataFrame(),
        "chart": pd.DataFrame(),
        "best": None,
        "best_label": None,
        "best_mape": None,
        "reliability": "غير محدد | N/A",
        "reliability_level": "unknown",
        "note": "",
    }

    if series.empty:
        out["note"] = "لا توجد بيانات كافية لهذا المؤشر."
        return out

    test_days = max(3, min(int(test_days), max(3, len(series) // 3)))
    if len(series) - test_days < min_train_days:
        out["note"] = (
            f"يلزم {min_train_days + test_days} يومًا على الأقل للاختبار؛ "
            f"المتوفر {len(series)} يومًا."
        )
        return out

    train = series.iloc[:-test_days]
    test = series.iloc[-test_days:]
    actual = test.to_numpy(dtype=float)

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    for key, model in MODELS.items():
        try:
            if key == "linear_seasonal":
                predicted = model(train, test_days, column)  # type: ignore[call-arg]
            else:
                predicted = model(train, test_days)  # type: ignore[call-arg]
        except Exception:  # noqa: S112
            # نموذج واحد فشل لا يجب أن يُسقِط الاختبار كله | One model failing must not abort the run
            continue
        predictions[key] = predicted
        row = {"model": key, "label": MODEL_LABELS.get(key, key)}
        row.update(_metrics(actual, predicted))
        rows.append(row)

    if not rows:
        out["note"] = "تعذّر تشغيل نماذج الاختبار."
        return out

    metrics = pd.DataFrame(rows)
    # الأفضل: أقل MAPE، ثم أقل RMSE
    metrics["_rank"] = metrics["mape"].fillna(metrics["rmse"])
    metrics = metrics.sort_values(["_rank", "rmse"]).drop(columns=["_rank"]).reset_index(drop=True)

    best_row = metrics.iloc[0]
    best = str(best_row["model"])
    label, level = reliability(best_row.get("mape"))

    chart = pd.DataFrame({
        "date": test.index,
        "actual": actual,
        "predicted": predictions.get(best, np.zeros(test_days)),
    })

    out.update({
        "available": True,
        "train_days": len(train),
        "metrics": metrics,
        "chart": chart,
        "best": best,
        "best_label": MODEL_LABELS.get(best, best),
        "best_mape": _num(best_row.get("mape")),
        "reliability": label,
        "reliability_level": level,
        "note": (
            f"تم التدريب على {len(train)} يومًا واختبار النموذج على {test_days} يومًا "
            "لم يرها. MAPE أقل = دقة أعلى."
        ),
    })
    return out


# ============================================================
# 3) محاكي السيناريوهات (What-if)
# ============================================================
def what_if(
    daily: pd.DataFrame,
    horizon_days: int = 30,
    window_days: int = 30,
    price_change_pct: float = 0.0,
    volume_change_pct: float = 0.0,
    expense_change_pct: float = 0.0,
    currency: str = "JOD",
) -> dict[str, Any]:
    """يحاكي أثر تغييرات السعر/الكمية/المصاريف على الإيراد والربح.

    الافتراض: تغيير السعر يؤثر على الإيراد، وتغيير الكمية يؤثر على الإيراد،
    وتغيير المصاريف يؤثر على إجمالي المصاريف. (بدون تفصيل تكاليف ثابتة/متغيرة)
    """
    out: dict[str, Any] = {
        "available": False, "currency": currency, "horizon_days": int(horizon_days),
        "window_days": int(window_days),
        "price_change_pct": float(price_change_pct) * 100,
        "volume_change_pct": float(volume_change_pct) * 100,
        "expense_change_pct": float(expense_change_pct) * 100,
        "baseline": {}, "scenario": {}, "delta": {}, "note": "",
    }
    if daily.empty or "revenue" not in daily.columns:
        out["note"] = "لا توجد بيانات إيراد كافية للمحاكاة."
        return out

    recent = daily.tail(max(7, min(int(window_days), len(daily))))
    base_revenue_daily = float(recent["revenue"].mean())
    base_expense_daily = float(recent["expense"].mean()) if "expense" in recent.columns else 0.0
    base_profit_daily = float(recent["profit"].mean()) if "profit" in recent.columns else base_revenue_daily - base_expense_daily

    horizon = max(1, int(horizon_days))
    base_revenue = base_revenue_daily * horizon
    base_expense = base_expense_daily * horizon
    base_profit = base_profit_daily * horizon

    price_factor = 1 + float(price_change_pct)
    volume_factor = 1 + float(volume_change_pct)
    expense_factor = 1 + float(expense_change_pct)

    scenario_revenue = base_revenue * price_factor * volume_factor
    scenario_expense = base_expense * expense_factor
    scenario_profit = scenario_revenue - scenario_expense

    def margin(profit: float, revenue: float) -> float | None:
        return _num(profit / revenue * 100) if revenue else None

    delta_profit = scenario_profit - base_profit
    delta_revenue = scenario_revenue - base_revenue
    delta_pct = (delta_profit / abs(base_profit) * 100) if abs(base_profit) > 1e-9 else None

    if delta_profit > 1e-9:
        verdict, level = "تحسّن | Better", "good"
    elif delta_profit < -1e-9:
        verdict, level = "تراجع | Worse", "bad"
    else:
        verdict, level = "بدون تغيير | Neutral", "ok"

    out.update({
        "available": True,
        "baseline": {
            "revenue": _num(base_revenue), "expense": _num(base_expense),
            "profit": _num(base_profit),
            "margin_pct": margin(base_profit, base_revenue),
        },
        "scenario": {
            "revenue": _num(scenario_revenue), "expense": _num(scenario_expense),
            "profit": _num(scenario_profit),
            "margin_pct": margin(scenario_profit, scenario_revenue),
        },
        "delta": {
            "revenue": _num(delta_revenue), "profit": _num(delta_profit),
            "profit_pct": _num(delta_pct),
        },
        "verdict": verdict,
        "level": level,
        "note": (
            f"محاكاة على متوسط آخر {len(recent)} يومًا، لأفق {horizon} يومًا. "
            "الافتراض أن المصاريف تتغير بنسبة مستقلة عن السعر/الكمية."
        ),
    })
    return out
