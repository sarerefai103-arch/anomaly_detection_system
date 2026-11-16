"""
data_loader.py
--------------
قراءة مرنة لمصادر البيانات + تقرير جودة البيانات.

يدعم: CSV (بفواصل/ترميزات مختلفة)، Excel (xlsx/xls)، JSON.
ويوفّر `data_quality()` لفحص النواقص والتكرارات والأعمدة المشبوهة.

المطوّر: sari refai
"""

from __future__ import annotations

import io
import json
import re
import warnings
from pathlib import Path

import pandas as pd

SUPPORTED_TYPES = ["csv", "xlsx", "xls", "json"]

_NUMERIC_CLEAN = re.compile(r"[^\d.\-+]")


# ============================================================
# 1) القراءة المرنة
# ============================================================
def _extension(name: str | None, fallback: str = "csv") -> str:
    suffix = Path(str(name or "")).suffix.lower().lstrip(".")
    return suffix if suffix in SUPPORTED_TYPES else fallback


def _read_bytes(source) -> bytes:
    data = source.read() if hasattr(source, "read") else Path(str(source)).read_bytes()
    if isinstance(data, str):
        return data.encode("utf-8")
    return data


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1256", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _guess_sep(sample: str) -> str:
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _read_csv(source) -> pd.DataFrame:
    text = _decode_text(_read_bytes(source))
    sample = "\n".join(text.splitlines()[:6])
    sep = _guess_sep(sample)
    return pd.read_csv(io.StringIO(text), sep=sep)


def _read_excel(source) -> pd.DataFrame:
    data = _read_bytes(source)
    return pd.read_excel(io.BytesIO(data))


def _read_json(source) -> pd.DataFrame:
    text = _decode_text(_read_bytes(source))
    obj = json.loads(text)
    if isinstance(obj, list):
        return pd.json_normalize(obj)
    if isinstance(obj, dict):
        # شائع: {"data": [...]}
        for key in ("data", "rows", "records", "items", "results"):
            if isinstance(obj.get(key), list):
                return pd.json_normalize(obj[key])
        try:
            return pd.json_normalize(obj)
        except Exception:
            return pd.DataFrame([obj])
    raise ValueError("صيغة JSON غير مدعومة | Unsupported JSON structure")


def read_dataset(source, name: str | None = None) -> tuple[pd.DataFrame, dict]:
    """يقرأ CSV/Excel/JSON ويعيد (DataFrame, معلومات المصدر)."""
    label = name or getattr(source, "name", None) or str(source)
    ext = _extension(label)

    if ext == "xlsx" or ext == "xls":
        df = _read_excel(source)
    elif ext == "json":
        df = _read_json(source)
    else:
        df = _read_csv(source)

    # تنظيف أساسي: أسماء أعمدة نصية بدون فراغات زائدة
    df.columns = [str(col).strip() for col in df.columns]

    meta = {
        "name": Path(str(label)).name,
        "extension": ext,
        "rows": len(df),
        "columns": int(df.shape[1]),
        "size_bytes": len(_read_bytes(source)) if not hasattr(source, "read") else None,
    }
    return df, meta


# ============================================================
# 2) جودة البيانات
# ============================================================
def _looks_numeric(series: pd.Series, sample: int = 300) -> bool:
    values = series.dropna().astype(str).head(sample)
    if values.empty:
        return False
    cleaned = values.map(lambda v: _NUMERIC_CLEAN.sub("", v.replace(",", "")))
    converted = pd.to_numeric(cleaned, errors="coerce")
    return float(converted.notna().mean()) >= 0.85


def _looks_date(series: pd.Series, sample: int = 200) -> bool:
    values = series.dropna().astype(str).head(sample)
    if values.empty:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(values, errors="coerce", format="mixed")
            if float(parsed.notna().mean()) < 0.85:
                alt = pd.to_datetime(values, errors="coerce", dayfirst=True, format="mixed")
                if float(alt.notna().mean()) > float(parsed.notna().mean()):
                    parsed = alt
    except Exception:
        return False
    return float(parsed.notna().mean()) >= 0.85


def data_quality(df: pd.DataFrame) -> dict:
    """يحسب مؤشر جودة (0-100) وقائمة ملاحظات قابلة للعرض."""
    rows, cols = df.shape
    total_cells = max(1, rows * cols)

    missing_by_col = df.isna().sum()
    missing_cells = int(missing_by_col.sum())
    missing_pct = missing_cells / total_cells * 100

    duplicates = int(df.duplicated().sum())
    duplicate_pct = duplicates / max(1, rows) * 100

    empty_cols = [str(c) for c in df.columns if df[c].isna().all()]
    constant_cols = [
        str(c) for c in df.columns
        if df[c].nunique(dropna=True) <= 1 and str(c) not in empty_cols
    ]

    numeric_as_text: list[str] = []
    date_as_text: list[str] = []
    mixed_types: list[str] = []

    for col in df.columns:
        series = df[col]
        if series.dtype == object or str(series.dtype) == "str":
            unique_types = {type(v).__name__ for v in series.dropna().head(200)}
            if len(unique_types) > 1:
                mixed_types.append(str(col))
            if _looks_numeric(series) and not _looks_date(series):
                numeric_as_text.append(str(col))
            elif _looks_date(series):
                date_as_text.append(str(col))

    negatives: dict[str, int] = {}
    for col in df.columns:
        if _looks_numeric(df[col]):
            values = pd.to_numeric(df[col], errors="coerce")
            count = int((values < 0).sum())
            if count:
                negatives[str(col)] = count

    # ترتيب الأعمدة حسب نسبة النواقص
    worst_missing = (
        missing_by_col[missing_by_col > 0]
        .sort_values(ascending=False)
        .head(8)
    )

    penalty = 0.0
    penalty += min(35.0, missing_pct * 1.4)
    penalty += min(20.0, duplicate_pct * 2.0)
    penalty += min(10.0, len(empty_cols) * 5.0)
    penalty += min(8.0, len(constant_cols) * 2.0)
    penalty += min(7.0, len(numeric_as_text) * 2.0)
    penalty += min(5.0, len(mixed_types) * 1.5)
    score = max(0, min(100, round(100 - penalty, 1)))

    if score >= 85:
        label = "ممتاز | Excellent"
        level = "good"
    elif score >= 70:
        label = "جيد | Good"
        level = "ok"
    elif score >= 50:
        label = "متوسط | Fair"
        level = "warn"
    else:
        label = "ضعيف | Poor"
        level = "bad"

    issues: list[dict] = []
    if missing_cells:
        issues.append({
            "severity": "warn" if missing_pct > 5 else "info",
            "issue_ar": "قيم ناقصة",
            "issue_en": "Missing values",
            "detail": f"{missing_cells:,} خلية ({missing_pct:.1f}%)",
        })
    if duplicates:
        issues.append({
            "severity": "warn" if duplicate_pct > 2 else "info",
            "issue_ar": "صفوف مكررة",
            "issue_en": "Duplicate rows",
            "detail": f"{duplicates:,} صف ({duplicate_pct:.1f}%)",
        })
    if empty_cols:
        issues.append({
            "severity": "bad", "issue_ar": "أعمدة فارغة تمامًا",
            "issue_en": "Empty columns", "detail": "، ".join(empty_cols[:6]),
        })
    if constant_cols:
        issues.append({
            "severity": "info", "issue_ar": "أعمدة ثابتة القيمة",
            "issue_en": "Constant columns", "detail": "، ".join(constant_cols[:6]),
        })
    if numeric_as_text:
        issues.append({
            "severity": "warn", "issue_ar": "أرقام مخزّنة كنص",
            "issue_en": "Numbers stored as text", "detail": "، ".join(numeric_as_text[:6]),
        })
    if date_as_text:
        issues.append({
            "severity": "info", "issue_ar": "تواريخ نصية (سيتم تحويلها)",
            "issue_en": "Text dates (auto-parsed)", "detail": "، ".join(date_as_text[:6]),
        })
    if mixed_types:
        issues.append({
            "severity": "warn", "issue_ar": "أنواع مختلطة بالعمود",
            "issue_en": "Mixed types", "detail": "، ".join(mixed_types[:6]),
        })
    if negatives:
        detail = "، ".join(f"{col}: {count:,}" for col, count in list(negatives.items())[:5])
        issues.append({
            "severity": "info", "issue_ar": "قيم سالبة",
            "issue_en": "Negative values", "detail": detail,
        })

    return {
        "score": score,
        "label": label,
        "level": level,
        "rows": rows,
        "columns": cols,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_pct, 2),
        "duplicates": duplicates,
        "duplicate_pct": round(duplicate_pct, 2),
        "empty_columns": empty_cols,
        "constant_columns": constant_cols,
        "numeric_as_text": numeric_as_text,
        "date_as_text": date_as_text,
        "mixed_types": mixed_types,
        "negatives": negatives,
        "worst_missing": {
            str(col): int(count) for col, count in worst_missing.items()
        },
        "issues": issues,
    }


def quality_label(score: float) -> str:
    if score >= 85:
        return "ممتاز | Excellent"
    if score >= 70:
        return "جيد | Good"
    if score >= 50:
        return "متوسط | Fair"
    return "ضعيف | Poor"
