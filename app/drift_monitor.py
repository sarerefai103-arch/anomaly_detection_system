"""
drift_monitor.py
----------------
مراقبة تغيّر البيانات (Data Drift) بين مجموعة مرجعية (Baseline) والبيانات الحالية.

المقاييس المعتمدة:
- PSI (Population Stability Index) لكل عمود:
    PSI < 0.10  → مستقر
    0.10–0.25   → مراقبة
    PSI > 0.25  → انحراف
- KS Test للأعمدة الرقمية و Chi-square للأعمدة الفئوية.
- Adversarial Validation: AUC لنموذج يحاول التمييز بين المرجع والحالي.
- Timeline: تتبّع متوسط PSI عبر النوافذ الزمنية.
- حماية من الأعمدة غير المناسبة: التواريخ والمعرّفات عالية التفرّد تُستبعد تلقائيًا.

المطوّر: sari refai
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

PSI_STABLE = 0.10
PSI_DRIFT = 0.25
MIN_ROWS = 20
MIN_WINDOW_ROWS = 20  # أقل عدد صفوف لنافذة زمنية موثوقة (نفس حد التحليل)
MISSING = "(ناقص | missing)"
OTHER = "(أخرى | other)"

STATUS_AR = {
    "stable": "مستقر",
    "watch": "مراقبة",
    "drift": "انحراف",
    "unknown": "غير محدد",
}
STATUS_EN = {
    "stable": "Stable",
    "watch": "Watch",
    "drift": "Drift",
    "unknown": "Unknown",
}
STATUS_COLORS = {
    "stable": "#00C853",
    "watch": "#FFB066",
    "drift": "#FF4D4D",
    "unknown": "#8A8A93",
}
LEVEL_AR = {
    "stable": "مستقر",
    "watch": "تغيّر متوسط",
    "drift": "تغيّر كبير",
}
LEVEL_EN = {
    "stable": "Stable",
    "watch": "Moderate Drift",
    "drift": "Significant Drift",
}
FREQ_AR = {"D": "يومي | Daily", "W": "أسبوعي | Weekly", "M": "شهري | Monthly"}

FEATURE_COLUMNS = [
    "feature", "kind", "psi", "status", "test", "statistic", "p_value",
    "ref_missing", "cur_missing", "missing_delta",
    "ref_mean", "cur_mean", "mean_shift_pct",
    "ref_std", "cur_std", "std_ratio",
    "ref_unique", "cur_unique", "new_categories_pct",
]


# ============================================================
# 1) أدوات اكتشاف الأنواع
# ============================================================
def parse_dates(series: pd.Series) -> pd.Series:
    """يحوّل عمودًا إلى تواريخ مع محاولة dayfirst عند الحاجة."""
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        if float(parsed.notna().mean()) < 0.85:
            alt = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
            if float(alt.notna().mean()) > float(parsed.notna().mean()):
                parsed = alt
    return parsed


def is_datetime(series: pd.Series) -> bool:
    """هل يبدو العمود تاريخًا؟"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    sample = series.dropna()
    if sample.empty:
        return False
    parsed = parse_dates(sample.head(500))
    return bool(parsed.notna().mean() >= 0.85)


def is_numeric(series: pd.Series) -> bool:
    """هل يبدو العمود رقميًا؟ (يتعامل مع الأرقام المخزّنة كنص والفاصلات)"""
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    sample = series.dropna()
    if sample.empty:
        return False
    cleaned = sample.head(500).astype(str).str.replace(",", "", regex=False)
    converted = pd.to_numeric(cleaned, errors="coerce")
    return bool(converted.notna().mean() >= 0.85)


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    """أعمدة التاريخ الصالحة للتحليل الزمني."""
    return [str(col) for col in df.columns if is_datetime(df[col])]


def detect_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(col) for col in df.columns if is_numeric(df[col])]


def as_text(series: pd.Series) -> pd.Series:
    """تحويل إلى نص مع تمييز القيم الناقصة (لتُحسب ضمن التوزيع)."""
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    s = s.reset_index(drop=True)
    return s.astype("object").where(s.notna(), MISSING).astype(str)


def classify_column(ref_col: pd.Series, cur_col: pd.Series, ref_len: int):
    """يعيد (النوع, سبب الاستبعاد). النوع: numeric / categorical / None."""
    if is_datetime(ref_col) or is_datetime(cur_col):
        return None, ("عمود تاريخ", "Date column")
    if is_numeric(ref_col) and is_numeric(cur_col):
        return "numeric", None
    unique = max(int(ref_col.nunique(dropna=True)), int(cur_col.nunique(dropna=True)))
    if unique > 50 and unique / max(1, ref_len) > 0.6:
        return None, ("معرّف عالي التفرّد", "High-cardinality identifier")
    return "categorical", None


# ============================================================
# 2) حساب PSI والاختبارات
# ============================================================
def psi_status(psi: float) -> str:
    if psi is None or not np.isfinite(psi):
        return "unknown"
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_DRIFT:
        return "watch"
    return "drift"


def stability_level(stability: float) -> str:
    if stability >= 75:
        return "stable"
    if stability >= 50:
        return "watch"
    return "drift"


def _num(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def psi_numeric(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI لعمود رقمي باستخدام حدود فئات مشتقة من المرجع (quantiles)."""
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(pd.Series(current), errors="coerce").dropna().to_numpy(dtype=float)
    if ref.size == 0 or cur.size == 0:
        return float("nan")

    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, int(bins) + 1)))
    if edges.size < 2:
        # مرجع ثابت: انحراف فقط إذا اختلفت القيم الحالية عنه
        return 0.0 if bool(np.allclose(cur, ref[0])) else 1.0

    inner = edges[1:-1]
    full_edges = np.concatenate(([-np.inf], inner, [np.inf]))
    ref_counts, _ = np.histogram(ref, bins=full_edges)
    cur_counts, _ = np.histogram(cur, bins=full_edges)
    return _psi_from_counts(ref_counts, cur_counts, eps)


def _psi_from_counts(ref_counts, cur_counts, eps: float = 1e-4) -> float:
    ref_counts = np.asarray(ref_counts, dtype=float)
    cur_counts = np.asarray(cur_counts, dtype=float)
    if ref_counts.size == 0 or cur_counts.size == 0:
        return float("nan")
    ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * ref_counts.size)
    cur_pct = (cur_counts + eps) / (cur_counts.sum() + eps * cur_counts.size)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _category_counts(ref_txt: pd.Series, cur_txt: pd.Series, top: int = 20):
    """تجميع أعلى الفئات من المرجع + سلة (أخرى) للفئات الجديدة."""
    counts = ref_txt.value_counts()
    categories = [str(cat) for cat in counts.head(top).index.tolist()]
    labels = [*categories, OTHER]
    ref_grouped = ref_txt.where(ref_txt.isin(categories), OTHER)
    cur_grouped = cur_txt.where(cur_txt.isin(categories), OTHER)
    ref_counts = ref_grouped.value_counts().reindex(labels, fill_value=0).to_numpy(dtype=float)
    cur_counts = cur_grouped.value_counts().reindex(labels, fill_value=0).to_numpy(dtype=float)
    return labels, ref_counts, cur_counts


def psi_categorical(reference, current, top: int = 20, eps: float = 1e-4) -> float:
    """PSI لعمود فئوي مع تجميع الفئات النادرة/الجديدة."""
    ref_txt = as_text(pd.Series(reference))
    cur_txt = as_text(pd.Series(current))
    if ref_txt.empty or cur_txt.empty:
        return float("nan")
    _labels, ref_counts, cur_counts = _category_counts(ref_txt, cur_txt, top)
    return _psi_from_counts(ref_counts, cur_counts, eps)


def new_category_share(reference, current) -> float:
    """نسبة قيم البيانات الحالية التي لم تظهر في المرجع."""
    ref_txt = as_text(pd.Series(reference))
    cur_txt = as_text(pd.Series(current))
    if cur_txt.empty:
        return float("nan")
    known = set(ref_txt.unique().tolist())
    return float((~cur_txt.isin(known)).mean())


# ============================================================
# 3) Adversarial Validation
# ============================================================
def adversarial_drift(reference: pd.DataFrame, current: pd.DataFrame,
                      features: list[str] | None = None,
                      max_rows: int = 3000) -> dict:
    """AUC لنموذج يحاول التمييز: 0.5 = لا انحراف، > 0.7 = انحراف واضح."""
    cols = [c for c in (features or list(reference.columns)) if c in current.columns]
    if not cols:
        return {"available": False, "auc": float("nan"), "level": "unknown",
                "note_ar": "لا توجد أعمدة مشتركة لفحص الفصل."}

    ref = reference[cols]
    cur = current[cols]
    total = len(ref) + len(cur)
    if total > max_rows:
        ratio_ref = len(ref) / max(1, total)
        n_ref = max(10, int(max_rows * ratio_ref))
        n_cur = max(10, max_rows - n_ref)
        ref = ref.sample(min(len(ref), n_ref), random_state=42)
        cur = cur.sample(min(len(cur), n_cur), random_state=42)

    combined = pd.concat([ref, cur], ignore_index=True)
    matrix = pd.DataFrame(index=range(len(combined)))
    for col in cols:
        series = combined[col]
        if is_numeric(series):
            matrix[col] = pd.to_numeric(series, errors="coerce").astype("float64")
        else:
            txt = as_text(series)
            matrix[col] = pd.Categorical(txt, categories=sorted(txt.unique())).codes.astype("float64")

    y = np.concatenate([np.zeros(len(ref)), np.ones(len(cur))])
    minority = int(min(y.sum(), len(y) - y.sum()))
    if minority < 30:
        return {"available": False, "auc": float("nan"), "level": "unknown",
                "note_ar": "عدد الصفوف غير كافٍ لفحص الفصل (يلزم 30 صفًا لكل مجموعة على الأقل)."}

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import (
            StratifiedGroupKFold,
            StratifiedKFold,
            cross_val_predict,
        )

        model = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, max_depth=3, random_state=42,
        )
        # الصفوف المتطابقة (مثل مقارنة لقطة محفوظة بنفس البيانات) تُجمَّع في الطية نفسها،
        # وإلا تتعارض التسمية بين النسختين فينقلب ترتيب الاحتمالات ويصبح AUC مضلّلًا.
        try:
            groups = matrix.round(6).astype(str).agg("|".join, axis=1).factorize()[0]
            folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
            proba = cross_val_predict(
                model, matrix, y, cv=folds, groups=groups, method="predict_proba",
            )[:, 1]
            strategy = "grouped"
        except Exception:
            groups = None
            strategy = "stratified"
            folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            proba = cross_val_predict(model, matrix, y, cv=folds, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, proba))
    except Exception as err:
        return {"available": False, "auc": float("nan"), "level": "unknown",
                "note_ar": f"تعذّر إجراء فحص الفصل: {type(err).__name__}"}

    if auc < 0.60:
        level = "stable"
        note = "النموذج لا يستطيع التمييز بين المرجع والحالي — التوزيعات متشابهة."
    elif auc < 0.70:
        level = "watch"
        note = "يوجد فصل بسيط بين المرجع والحالي — يستحق المراقبة."
    else:
        level = "drift"
        note = "النموذج يميّز المرجع عن الحالي بسهولة — يوجد تغيّر هيكلي في البيانات."
    return {
        "available": True,
        "auc": round(auc, 4),
        "level": level,
        "label_ar": LEVEL_AR[level],
        "note_ar": note,
        "rows_used": len(combined),
        "features_used": len(cols),
        "cv": strategy,
        "duplicate_rows": int(len(combined) - len(np.unique(groups))) if groups is not None else 0,
    }


# ============================================================
# 4) التحليل الكامل
# ============================================================
def analyze_drift(reference: pd.DataFrame, current: pd.DataFrame, *,
                  bins: int = 10, max_features: int = 60,
                  include_adversarial: bool = True) -> dict:
    """يقارن مجموعة مرجعية بمجموعة حالية ويعيد تقريرًا كاملًا."""
    ref = reference.copy()
    cur = current.copy()
    common = [str(col) for col in ref.columns if col in cur.columns]
    bins = int(max(4, min(30, bins)))

    records: list[dict] = []
    skipped: list[dict] = []

    for col in common:
        if len(records) >= max_features:
            skipped.append({
                "feature": col,
                "reason_ar": "تجاوز الحد الأقصى للأعمدة",
                "reason_en": "Feature limit reached",
            })
            continue
        ref_col, cur_col = ref[col], cur[col]
        kind, reason = classify_column(ref_col, cur_col, len(ref))
        if kind is None:
            skipped.append({
                "feature": col,
                "reason_ar": reason[0],
                "reason_en": reason[1],
            })
            continue

        if kind == "numeric":
            ref_values = pd.to_numeric(ref_col, errors="coerce").dropna()
            cur_values = pd.to_numeric(cur_col, errors="coerce").dropna()
            if len(ref_values) < MIN_ROWS or len(cur_values) < MIN_ROWS:
                skipped.append({
                    "feature": col,
                    "reason_ar": "قيم رقمية غير كافية",
                    "reason_en": "Insufficient numeric values",
                })
                continue
            psi = psi_numeric(ref_values, cur_values, bins)
            statistic, p_value = stats.ks_2samp(ref_values, cur_values)
            test = "KS"
            ref_mean = _num(ref_values.mean())
            cur_mean = _num(cur_values.mean())
            if abs(ref_mean) > 1e-12:
                mean_shift = (cur_mean - ref_mean) / abs(ref_mean) * 100.0
            else:
                mean_shift = float("nan")
            ref_std = _num(ref_values.std())
            cur_std = _num(cur_values.std())
            std_ratio = cur_std / ref_std if ref_std > 1e-12 else float("nan")
            new_cats = float("nan")
        else:
            ref_txt, cur_txt = as_text(ref_col), as_text(cur_col)
            if len(ref_txt) < MIN_ROWS or len(cur_txt) < MIN_ROWS:
                skipped.append({
                    "feature": col,
                    "reason_ar": "صفوف غير كافية",
                    "reason_en": "Insufficient rows",
                })
                continue
            _labels, ref_counts, cur_counts = _category_counts(ref_txt, cur_txt)
            psi = _psi_from_counts(ref_counts, cur_counts)
            table = np.vstack([ref_counts, cur_counts])
            keep = table.sum(axis=0) > 0
            table = table[:, keep]
            if table.shape[1] >= 2:
                try:
                    statistic, p_value, _dof, _expected = stats.chi2_contingency(table)
                except ValueError:
                    statistic, p_value = float("nan"), float("nan")
            else:
                statistic, p_value = float("nan"), float("nan")
            test = "Chi²"
            ref_mean = cur_mean = mean_shift = float("nan")
            ref_std = cur_std = std_ratio = float("nan")
            new_cats = new_category_share(ref_txt, cur_txt)

        records.append({
            "feature": col,
            "kind": kind,
            "psi": _num(psi),
            "status": psi_status(psi),
            "test": test,
            "statistic": _num(statistic),
            "p_value": _num(p_value),
            "ref_missing": float(ref_col.isna().mean()),
            "cur_missing": float(cur_col.isna().mean()),
            "missing_delta": float(cur_col.isna().mean() - ref_col.isna().mean()),
            "ref_mean": ref_mean,
            "cur_mean": cur_mean,
            "mean_shift_pct": _num(mean_shift),
            "ref_std": ref_std,
            "cur_std": cur_std,
            "std_ratio": _num(std_ratio),
            "ref_unique": int(ref_col.nunique(dropna=True)),
            "cur_unique": int(cur_col.nunique(dropna=True)),
            "new_categories_pct": _num(new_cats),
        })

    features = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    if not features.empty:
        features = (
            features.sort_values("psi", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

    valid = features["psi"].dropna() if not features.empty else pd.Series(dtype=float)
    feature_count = len(features)
    if len(valid):
        scaled = np.clip(valid.to_numpy(dtype=float) / 0.5, 0.0, 1.0)
        drift_count_pre = int((valid > PSI_DRIFT).sum())
        drift_share_pre = drift_count_pre / max(1, feature_count)
        # مزيج من متوسط الانحراف وأقصى انحراف ونسبة الأعمدة المنحرفة
        drift_index = 100.0 * (
            0.40 * float(np.mean(scaled))
            + 0.45 * float(np.max(scaled))
            + 0.15 * drift_share_pre
        )
        stability = round(100.0 - drift_index, 1)
        max_row = features.loc[valid.idxmax()]
        max_feature = str(max_row["feature"])
        max_psi = round(float(valid.max()), 4)
        avg_psi = round(float(valid.mean()), 4)
    else:
        stability = 0.0
        max_feature = "—"
        max_psi = 0.0
        avg_psi = 0.0

    status_counts = features["status"].value_counts() if not features.empty else pd.Series(dtype=int)
    level = stability_level(stability)
    drift_count = int(status_counts.get("drift", 0))
    watch_count = int(status_counts.get("watch", 0))
    stable_count = int(status_counts.get("stable", 0))

    summary = {
        "stability": stability,
        "level": level,
        "label_ar": LEVEL_AR[level],
        "label_en": LEVEL_EN[level],
        "feature_count": feature_count,
        "stable_count": stable_count,
        "watch_count": watch_count,
        "drift_count": drift_count,
        "avg_psi": avg_psi,
        "max_psi": max_psi,
        "max_feature": max_feature,
        "drift_share": round(drift_count / feature_count * 100.0, 1) if feature_count else 0.0,
        "rows_ref": len(ref),
        "rows_cur": len(cur),
        "columns_ref": int(ref.shape[1]),
        "columns_cur": int(cur.shape[1]),
        "common_columns": len(common),
        "bins": bins,
    }

    analyzed = features["feature"].tolist() if not features.empty else []
    if include_adversarial and analyzed:
        adversarial = adversarial_drift(ref, cur, features=analyzed)
    else:
        adversarial = {
            "available": False,
            "auc": float("nan"),
            "level": "unknown",
            "note_ar": "فحص الفصل الذكي غير مُفعّل.",
        }

    return {
        "features": features,
        "summary": summary,
        "adversarial": adversarial,
        "skipped": skipped,
        "bins": bins,
    }


# ============================================================
# 5) الخط الزمني للانحراف
# ============================================================
def _period_label(period: pd.Period, freq: str) -> str:
    if freq == "M":
        return f"{period.year}-{period.month:02d}"
    return f"{period.start_time:%Y-%m-%d}"


def timeline_matrix(reference: pd.DataFrame, current: pd.DataFrame, date_col: str, *,
                    bins: int = 10, freq: str | None = None,
                    max_features: int = 12) -> dict:
    """متوسط PSI عبر نوافذ زمنية + مصفوفة (عمود × نافذة) للخريطة الحرارية."""
    if date_col not in current.columns:
        return {
            "available": False,
            "note_ar": "عمود التاريخ غير موجود في البيانات الحالية.",
        }

    dates = parse_dates(current[date_col])
    mask = dates.notna()
    if int(mask.sum()) < MIN_ROWS:
        return {
            "available": False,
            "note_ar": "عدد التواريخ الصالحة غير كافٍ لبناء خط زمني (يلزم 20 على الأقل).",
        }

    cur = current.loc[mask].copy()
    cur["__drift_date"] = dates[mask].to_numpy()
    cur = cur.sort_values("__drift_date")
    span_days = int((cur["__drift_date"].max() - cur["__drift_date"].min()).days)
    if span_days < 1:
        return {
            "available": False,
            "note_ar": "كل الصفوف في يوم واحد — لا يمكن بناء خط زمني.",
        }

    if freq not in ("D", "W", "M"):
        freq = "D" if span_days <= 45 else ("W" if span_days <= 180 else "M")
    cur["__period"] = cur["__drift_date"].dt.to_period(freq)

    pairs: list[tuple[str, str]] = []
    for raw_col in reference.columns:
        col = str(raw_col)
        if col not in cur.columns or col == str(date_col):
            continue
        kind, _reason = classify_column(reference[col], cur[col], len(reference))
        if kind:
            pairs.append((col, kind))
    if not pairs:
        return {
            "available": False,
            "note_ar": "لا توجد أعمدة قابلة للتحليل الزمني.",
        }

    kind_map = dict(pairs)
    overall: dict[str, float] = {}
    for col, kind in pairs:
        if kind == "numeric":
            overall[col] = psi_numeric(reference[col], cur[col], bins)
        else:
            overall[col] = psi_categorical(reference[col], cur[col])
    ranked = sorted(
        [col for col, value in overall.items() if np.isfinite(value)],
        key=lambda col: overall[col],
        reverse=True,
    )
    selected = ranked[:max_features] or [col for col, _kind in pairs][:max_features]

    periods = list(cur["__period"].drop_duplicates().sort_values())
    rows: list[dict] = []
    matrix: dict[str, list[float]] = {col: [] for col in selected}
    for period in periods:
        chunk = cur[cur["__period"] == period]
        chunk_rows = len(chunk)
        # نوافذ صغيرة لا تتحمّل 10 فئات: نقلّل الفئات تلقائيًا لتجنّب تضخّم PSI
        window_bins = int(max(3, min(bins, chunk_rows // 10)))
        window_top = int(max(3, min(20, chunk_rows // 6)))
        reliable = chunk_rows >= MIN_WINDOW_ROWS
        values: list[float] = []
        for col in selected:
            if kind_map[col] == "numeric":
                value = psi_numeric(reference[col], chunk[col], window_bins)
            else:
                value = psi_categorical(reference[col], chunk[col], window_top)
            matrix[col].append(_num(value))
            if np.isfinite(value):
                values.append(float(value))
        rows.append({
            "period": str(period),
            "label": _period_label(period, freq),
            "rows": chunk_rows,
            "avg_psi": float(np.mean(values)) if values else float("nan"),
            "max_psi": float(np.max(values)) if values else float("nan"),
            "drift_count": int(sum(1 for value in values if value > PSI_DRIFT)),
            "drift_share": float(
                sum(1 for value in values if value > PSI_DRIFT) / len(values)
            ) if values else float("nan"),
            "bins_used": window_bins,
            "top_used": window_top,
            "reliable": bool(reliable),
        })

    timeline = pd.DataFrame(rows)
    labels = timeline["label"].tolist()
    matrix_df = pd.DataFrame({col: matrix[col] for col in selected}, index=labels)

    unreliable = int((~timeline["reliable"]).sum())
    if unreliable:
        note = (
            f"المرجع ثابت، وكل نقطة تمثل نافذة من البيانات الحالية. "
            f"{unreliable} من {len(timeline)} نافذة تحتوي أقل من {MIN_WINDOW_ROWS} صفًا "
            f"(مؤشرها استرشادي فقط)."
        )
    else:
        note = "المرجع ثابت، وكل نقطة تمثل نافذة من البيانات الحالية."

    return {
        "available": True,
        "freq": freq,
        "freq_label_ar": FREQ_AR[freq],
        "date_col": str(date_col),
        "timeline": timeline,
        "matrix": matrix_df.T,
        "selected_features": selected,
        "reliable_windows": int(timeline["reliable"].sum()) if len(timeline) else 0,
        "unreliable_windows": unreliable,
        "note_ar": note,
    }


# ============================================================
# 6) التوصيات والتصدير
# ============================================================
def build_recommendations(report: dict, timeline: dict | None = None) -> list[dict]:
    """توصيات عربية مبنية على قواعد واضحة."""
    features = report.get("features")
    if features is None or features.empty:
        return [{
            "icon": "ℹ️",
            "title_ar": "لا توجد أعمدة قابلة للمقارنة",
            "title_en": "No comparable columns",
            "body_ar": "تأكد أن ملف المرجع والبيانات الحالية يشتركان في أسماء الأعمدة نفسها.",
        }]

    recs: list[dict] = []
    drift = features[features["status"] == "drift"]
    watch = features[features["status"] == "watch"]

    if not drift.empty:
        names = "، ".join(
            f"{row.feature} ({row.psi:.2f})" for row in drift.head(3).itertuples()
        )
        recs.append({
            "icon": "🚨",
            "title_ar": "أعمدة تحتاج تحقيقًا فوريًا",
            "title_en": "Investigate drifted columns now",
            "body_ar": (
                f"تغيّر توزيع {len(drift)} عمود بوضوح (PSI > {PSI_DRIFT:.2f}): {names}. "
                "افحص مصدر البيانات وخطوات التحويل والفلترة قبل استخدام النتائج في قرارات أو نماذج."
            ),
        })
    elif not watch.empty:
        names = "، ".join(
            f"{row.feature} ({row.psi:.2f})" for row in watch.head(3).itertuples()
        )
        recs.append({
            "icon": "🟠",
            "title_ar": "تغيّرات مبكّرة تستحق المراقبة",
            "title_en": "Early changes worth watching",
            "body_ar": (
                f"{len(watch)} عمود في نطاق المراقبة (0.10–0.25): {names}. "
                "لا حاجة لتدخّل عاجل، لكن راقبها أسبوعيًا قبل أن تتجاوز حدّ الانحراف."
            ),
        })
    else:
        recs.append({
            "icon": "✅",
            "title_ar": "التوزيعات مستقرة",
            "title_en": "Distributions are stable",
            "body_ar": (
                "لا يوجد عمود تجاوز حدّ المراقبة. استمر في المقارنة الدورية مع كل تحديث للبيانات."
            ),
        })

    if "new_categories_pct" in features.columns:
        new = features[features["new_categories_pct"] > 0.05]
        if not new.empty:
            names = "، ".join(
                f"{row.feature} ({row.new_categories_pct * 100:.1f}%)"
                for row in new.head(3).itertuples()
            )
            recs.append({
                "icon": "🆕",
                "title_ar": "فئات جديدة في البيانات",
                "title_en": "New categories detected",
                "body_ar": (
                    f"{names}. حدّث قوائم الفئات والترميز (Encoding) في النماذج والتقارير "
                    "حتى لا تُعامل القيم الجديدة كقيم مجهولة."
                ),
            })

    if "missing_delta" in features.columns:
        missing = features[features["missing_delta"].abs() > 0.05]
        if not missing.empty:
            names = "، ".join(
                f"{row.feature} ({row.missing_delta * 100:+.1f} نقطة)"
                for row in missing.head(3).itertuples()
            )
            recs.append({
                "icon": "🕳️",
                "title_ar": "تغيّر في نسب القيم الناقصة",
                "title_en": "Missing-value rate changed",
                "body_ar": (
                    f"{names}. راجع جمع البيانات أو الربط بين الجداول؛ ارتفاع النواقص قد يكون سببًا "
                    "لانحراف التوزيع وليس مجرد مشكلة جودة."
                ),
            })

    adversarial = report.get("adversarial") or {}
    if adversarial.get("available") and adversarial.get("auc", 0) >= 0.70:
        recs.append({
            "icon": "🧠",
            "title_ar": "أعد تدريب النماذج",
            "title_en": "Retrain affected models",
            "body_ar": (
                f"قدرة النموذج على فصل المرجع عن الحالي عالية (AUC = {adversarial['auc']:.2f}). "
                "أعد تدريب نماذج كشف الانحرافات/التوقّع على بيانات حديثة، وقارن الأداء قبل وبعد."
            ),
        })

    if timeline and timeline.get("available"):
        series = timeline.get("timeline")
        total_windows = len(series) if series is not None else 0
        if series is not None and total_windows >= 3:
            reliable = series[series["reliable"]] if "reliable" in series.columns else series
            valid = reliable["avg_psi"].dropna().to_numpy(dtype=float)
            # الاتجاه يُحسب على النوافذ الموثوقة فقط (>= 30 صفًا) لتفادي ضجيج النوافذ الصغيرة
            if valid.size >= 3:
                x = np.arange(valid.size, dtype=float)
                delta = float(valid[-1] - valid[0])
                slope = float(np.polyfit(x, valid, 1)[0])
                last_avg = float(valid[-1])
                if delta > 0.05 and slope > 0 and last_avg >= PSI_STABLE:
                    recs.append({
                        "icon": "📈",
                        "title_ar": "اتجاه تصاعدي للانحراف",
                        "title_en": "Drift is trending upward",
                        "body_ar": (
                            f"ارتفع متوسط PSI بمقدار {delta:.3f} بين أول وآخر نافذة موثوقة "
                            f"({valid.size} نافذة). حدّد التاريخ الذي بدأ فيه التغيّر واربطه "
                            "بإطلاق منتج أو تغيير في الحملات أو المصدر."
                        ),
                    })

        unreliable = int(timeline.get("unreliable_windows") or 0)
        if unreliable:
            recs.append({
                "icon": "🧩",
                "title_ar": "نوافذ زمنية صغيرة",
                "title_en": "Small time windows",
                "body_ar": (
                    f"{unreliable} من {total_windows} نافذة تحتوي أقل من {MIN_WINDOW_ROWS} صفًا، "
                    "لذا مؤشراتها استرشادية فقط ولا تُستخدم في حساب الاتجاه. "
                    "اختر تجميعًا أعرض (شهري) أو وسّع البيانات لنتائج أدق."
                ),
            })

    return recs


def report_csv(report: dict) -> str:
    """تقرير CSV قابل للفتح في Excel."""
    features = report.get("features")
    if features is None or features.empty:
        return "feature,kind,psi,status\n"
    table = features.copy()
    for col in ("psi", "p_value", "mean_shift_pct", "missing_delta", "new_categories_pct"):
        if col in table.columns:
            table[col] = table[col].round(6)
    return table.to_csv(index=False, na_rep="")


def summary_markdown(report: dict, ref_label: str = "", cur_label: str = "") -> str:
    """ملخص Markdown جاهز للتنزيل أو اللصق في التقرير."""
    summary = report["summary"]
    adversarial = report.get("adversarial") or {}
    lines = [
        "# تقرير مراقبة تغيّر البيانات | Data Drift Report",
        "",
        f"- **المرجع | Reference:** {ref_label or '—'} — {summary['rows_ref']:,} صف",
        f"- **الحالي | Current:** {cur_label or '—'} — {summary['rows_cur']:,} صف",
        f"- **وقت التقرير | Generated:** {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        "",
        "## الملخص | Summary",
        "",
        f"- **مؤشر الاستقرار | Stability Score:** {summary['stability']:.1f}/100 ({summary['label_ar']} | {summary['label_en']})",
        (
            f"- **الأعمدة المحلّلة | Analyzed features:** {summary['feature_count']} "
            f"(مستقر {summary['stable_count']} · مراقبة {summary['watch_count']} · "
            f"انحراف {summary['drift_count']})"
        ),
        f"- **متوسط PSI | Average PSI:** {summary['avg_psi']:.3f}",
        f"- **أعلى PSI | Max PSI:** {summary['max_psi']:.3f} ({summary['max_feature']})",
    ]
    if adversarial.get("available"):
        lines.append(f"- **Adversarial AUC:** {adversarial['auc']:.3f}")
    lines += [
        "",
        "## أعلى الأعمدة تأثرًا | Top Drifted Features",
        "",
        "| العمود | النوع | PSI | الحالة | الاختبار | p-value |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["features"].head(15).itertuples():
        p_value = "—" if not np.isfinite(row.p_value) else f"{row.p_value:.3g}"
        kind = "رقمي | Numeric" if row.kind == "numeric" else "فئوي | Categorical"
        lines.append(
            f"| {row.feature} | {kind} | {row.psi:.3f} | "
            f"{STATUS_AR[row.status]} | {row.test} | {p_value} |"
        )
    lines += ["", "## التوصيات | Recommendations", ""]
    for rec in build_recommendations(report):
        lines.append(f"- **{rec['title_ar']} | {rec['title_en']}:** {rec['body_ar']}")
    lines += [
        "",
        "> الحدود: PSI < 0.10 مستقر · 0.10–0.25 مراقبة · > 0.25 انحراف.",
        "> PSI measures how much a feature's distribution changed versus the baseline.",
    ]
    return "\n".join(lines) + "\n"
