"""اختبارات وحدة لمحرك Drift — تعمل مع pytest أو مباشرة.

    python tests/test_drift_unit.py
    pytest tests/test_drift_unit.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import numpy as np
import pandas as pd

import drift_monitor as dm

rng = np.random.default_rng(42)
N = 800


def make_frame(shift=0.0, city_p=None, missing=0.0, dates=None, n=N):
    day_index = dates if dates is not None else pd.date_range("2026-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "amount": rng.normal(100 + shift, 12, n),
        "qty": rng.integers(1, 6, n).astype(float),
        "city": rng.choice(["Amman", "Irbid", "Zarqa", "Aqaba"], n, p=city_p),
        "status": rng.choice(["Completed", "Pending", "Cancelled"], n, p=[.7, .2, .1]),
        "date": pd.Index(day_index).astype(str),
        "client_id": [f"CUST-{i:05d}" for i in range(n)],
        "constant": 7.0,
    })
    if missing:
        idx = rng.choice(n, int(n * missing), replace=False)
        df.loc[idx, "amount"] = np.nan
    return df


# ---------------- PSI ----------------
def test_psi_numeric_identical():
    ref = pd.Series(rng.normal(50, 5, 500))
    assert dm.psi_numeric(ref, ref) < 1e-6, "identical numeric PSI must be ~0"


def test_psi_numeric_shift():
    ref = pd.Series(rng.normal(50, 5, 500))
    cur = pd.Series(rng.normal(80, 5, 500))
    value = dm.psi_numeric(ref, cur)
    assert value > dm.PSI_DRIFT, f"shifted numeric PSI {value} must exceed {dm.PSI_DRIFT}"


def test_psi_numeric_constant():
    ref = pd.Series([5.0] * 100)
    assert dm.psi_numeric(ref, pd.Series([5.0] * 60)) == 0.0
    assert dm.psi_numeric(ref, pd.Series([6.0] * 60)) == 1.0


def test_psi_numeric_empty():
    assert np.isnan(dm.psi_numeric(pd.Series([], dtype=float), pd.Series([1.0, 2.0])))


def test_psi_categorical_identical():
    ref = pd.Series(rng.choice(list("ABCD"), 400, p=[.4, .3, .2, .1]))
    assert dm.psi_categorical(ref, ref) < 1e-6


def test_psi_categorical_shift():
    ref = pd.Series(rng.choice(["A", "B"], 400, p=[.9, .1]))
    cur = pd.Series(rng.choice(["A", "B"], 400, p=[.2, .8]))
    assert dm.psi_categorical(ref, cur) > dm.PSI_DRIFT


def test_new_category_share():
    ref = pd.Series(["A"] * 90 + ["B"] * 10)
    cur = pd.Series(["A"] * 50 + ["C"] * 50)
    assert abs(dm.new_category_share(ref, cur) - 0.5) < 1e-9


# ---------------- type detection ----------------
def test_detect_dates():
    df = make_frame()
    cols = dm.detect_date_columns(df)
    assert cols == ["date"], f"unexpected date columns {cols}"


def test_detect_numeric():
    df = make_frame()
    cols = dm.detect_numeric_columns(df)
    assert set(cols) == {"amount", "qty", "constant"}, cols


def test_parse_dates_dayfirst():
    parsed = dm.parse_dates(pd.Series(["01/02/2026", "03/04/2026", "05/06/2026"]))
    assert parsed.notna().all(), "dayfirst fallback failed"


def test_numeric_as_text():
    assert dm.is_numeric(pd.Series(["1,200", "350", "40"])), "comma numbers must be numeric"
    assert not dm.is_numeric(pd.Series(["Amman", "Irbid", "Zarqa"]))


# ---------------- analyze_drift ----------------
def test_analyze_stable():
    ref = make_frame()
    cur = make_frame()
    report = dm.analyze_drift(ref, cur, bins=10)
    summary = report["summary"]
    assert report["features"].shape[0] == 5, f"expected 5 features, got {report['features']['feature'].tolist()}"
    assert summary["level"] == "stable", summary
    assert summary["stability"] >= 75, summary["stability"]
    assert summary["drift_count"] == 0, summary
    assert set(report["features"]["feature"]) == {"amount", "qty", "city", "status", "constant"}


def test_analyze_exclusions():
    report = dm.analyze_drift(make_frame(), make_frame(), bins=10)
    skipped = {row["feature"]: row["reason_en"] for row in report["skipped"]}
    assert skipped.get("date") == "Date column", skipped
    assert skipped.get("client_id") == "High-cardinality identifier", skipped


def test_analyze_drifted_numeric():
    report = dm.analyze_drift(make_frame(), make_frame(shift=70), bins=10)
    row = report["features"].set_index("feature").loc["amount"]
    assert row["status"] == "drift", row.to_dict()
    assert row["psi"] > dm.PSI_DRIFT
    assert row["test"] == "KS"
    assert report["summary"]["level"] == "drift", report["summary"]
    assert report["summary"]["max_feature"] == "amount"


def test_analyze_drifted_categorical():
    report = dm.analyze_drift(make_frame(), make_frame(city_p=[.05, .05, .05, .85]), bins=10)
    row = report["features"].set_index("feature").loc["city"]
    assert row["status"] == "drift", row.to_dict()
    assert row["test"] == "Chi²"
    assert 0 <= row["p_value"] < 0.01


def test_analyze_missing_shift():
    report = dm.analyze_drift(make_frame(), make_frame(missing=0.3), bins=10)
    row = report["features"].set_index("feature").loc["amount"]
    assert abs(row["missing_delta"] - 0.3) < 0.06, row["missing_delta"]
    assert row["ref_missing"] == 0.0


def test_analyze_identical_zero():
    df = make_frame()
    report = dm.analyze_drift(df, df.copy(), bins=8)
    assert (report["features"]["psi"] < 1e-9).all()
    assert report["summary"]["stability"] == 100.0, report["summary"]


def test_analyze_bins_clamped():
    report = dm.analyze_drift(make_frame(), make_frame(), bins=500)
    assert report["bins"] == 30, report["bins"]
    report = dm.analyze_drift(make_frame(), make_frame(), bins=1)
    assert report["bins"] == 4, report["bins"]


def test_analyze_small():
    small = make_frame().head(10)
    report = dm.analyze_drift(small, small.copy(), bins=10)
    assert report["features"].empty
    assert report["summary"]["stability"] == 0.0
    reasons = {row["reason_en"] for row in report["skipped"]}
    assert "Insufficient numeric values" in reasons, reasons


def test_analyze_empty():
    report = dm.analyze_drift(pd.DataFrame(), pd.DataFrame(), include_adversarial=False)
    assert list(report["features"].columns) == dm.FEATURE_COLUMNS
    assert report["summary"]["feature_count"] == 0
    assert report["summary"]["max_feature"] == "—"
    assert report["adversarial"]["available"] is False


def test_analyze_no_common_columns():
    left = pd.DataFrame({"a": rng.normal(0, 1, 60)})
    right = pd.DataFrame({"b": rng.normal(0, 1, 60)})
    report = dm.analyze_drift(left, right, include_adversarial=False)
    assert report["features"].empty
    assert report["summary"]["common_columns"] == 0


def test_analyze_sorted():
    report = dm.analyze_drift(make_frame(), make_frame(shift=40), bins=10)
    values = report["features"]["psi"].tolist()
    assert values == sorted(values, reverse=True), "features must be sorted by PSI desc"


def test_analyze_feature_limit():
    wide = pd.DataFrame({f"f{i}": rng.normal(0, 1, 60) for i in range(12)})
    report = dm.analyze_drift(wide, wide.copy(), bins=10, max_features=5, include_adversarial=False)
    assert report["features"].shape[0] == 5
    assert sum(1 for row in report["skipped"] if row["reason_en"] == "Feature limit reached") == 7


# ---------------- adversarial ----------------
def test_adversarial_stable():
    a = make_frame()
    result = dm.adversarial_drift(a, make_frame())
    assert result["available"] is True, result
    assert result["auc"] < 0.60, result["auc"]
    assert result["level"] == "stable", result


def test_adversarial_drift():
    result = dm.adversarial_drift(make_frame(), make_frame(shift=80))
    assert result["available"] is True, result
    assert result["auc"] > 0.70, result["auc"]
    assert result["level"] == "drift", result


def test_adversarial_small():
    a = make_frame().head(10)
    result = dm.adversarial_drift(a, a.copy())
    assert result["available"] is False, result


def test_adversarial_identical_auc_half():
    """نفس البيانات بالضبط => AUC ≈ 0.5 (تجميع الصفوف المتطابقة في طية واحدة)."""
    df = make_frame()
    result = dm.adversarial_drift(df, df.copy())
    assert result["available"] is True, result
    assert result["cv"] == "grouped", result
    assert result["duplicate_rows"] > 0, result
    assert abs(result["auc"] - 0.5) <= 0.02, result["auc"]
    assert result["level"] == "stable", result


def test_adversarial_duplicate_noise_auc_half():
    """بنية تكرارية بلا إشارة => يجب ألا ينقلب المؤشر."""
    block = pd.DataFrame({c: rng.normal(0, 1, 200) for c in ("a", "b", "c")})
    doubled = pd.concat([block, block], ignore_index=True)
    result = dm.adversarial_drift(block, doubled)
    assert result["available"] is True, result
    assert 0.35 <= result["auc"] <= 0.65, result["auc"]


def test_adversarial_numeric_shift_full():
    df = make_frame()
    result = dm.adversarial_drift(df, df.assign(amount=df["amount"] * 2.2))
    assert result["available"] is True, result
    assert result["auc"] > 0.95, result["auc"]
    assert result["level"] == "drift", result


def test_adversarial_categorical_shift():
    df = make_frame()
    # عمود واحد ينتقل من خليط إلى فئة واحدة => فصل جزئي مرتفع (بقية الأعمدة متطابقة)
    result = dm.adversarial_drift(df, df.assign(status="Completed"))
    assert result["available"] is True, result
    assert result["auc"] > 0.60, result["auc"]
    assert result["level"] in ("watch", "drift"), result
    # انحراف فئوي كامل في عمودين => فصل شبه تام
    strong = dm.adversarial_drift(df, df.assign(status="Pending", city="Salt"))
    assert strong["auc"] > 0.95, strong["auc"]
    assert strong["level"] == "drift", strong


# ---------------- timeline ----------------
def test_timeline_daily():
    dates = pd.date_range("2026-01-01", periods=N, freq="D")
    report = dm.timeline_matrix(make_frame(dates=dates), make_frame(shift=30, dates=dates), "date", bins=10)
    assert report["available"] is True, report
    assert report["freq"] == "M", report["freq"]  # 800 يومًا > 180 => شهري تلقائي
    assert len(report["selected_features"]) >= 1
    assert report["matrix"].shape[0] == len(report["selected_features"])
    assert {"avg_psi", "max_psi", "reliable", "bins_used", "top_used"} <= set(report["timeline"].columns)
    assert report["timeline"]["rows"].sum() == N


def test_timeline_auto_weekly():
    dates = pd.date_range("2026-01-01", periods=120, freq="D")
    report = dm.timeline_matrix(make_frame(n=120), make_frame(n=120, dates=dates), "date", bins=10)
    assert report["available"] is True
    assert report["freq"] == "W", report["freq"]


def test_timeline_monthly_all_reliable():
    dates = pd.date_range("2026-01-01", periods=N, freq="D")
    report = dm.timeline_matrix(make_frame(dates=dates), make_frame(dates=dates), "date", bins=10, freq="M")
    # فقط النافذة الأخيرة الجزئية قد تقل عن الحد
    timeline = report["timeline"]
    full_months = timeline.iloc[:-1]
    assert full_months["reliable"].all(), timeline[["label", "rows", "reliable"]]
    assert report["unreliable_windows"] <= 1, report["unreliable_windows"]
    assert "نافذة" in report["note_ar"] or report["unreliable_windows"] == 0


def test_timeline_small_windows_flagged():
    dates = pd.date_range("2026-01-01", periods=40, freq="h")
    frame = make_frame(n=40, dates=dates)
    report = dm.timeline_matrix(frame, frame.copy(), "date", bins=10, freq="D")
    assert report["available"] is True
    assert report["unreliable_windows"] > 0
    assert "صفًا" in report["note_ar"]


def test_timeline_single_day():
    frame = pd.DataFrame({"x": rng.normal(0, 1, 40), "day": ["2026-01-01"] * 40})
    report = dm.timeline_matrix(frame, frame.copy(), "day")
    assert report["available"] is False, report


def test_timeline_missing_date_col():
    report = dm.timeline_matrix(make_frame(), make_frame(), "nope")
    assert report["available"] is False
    assert "غير موجود" in report["note_ar"]


def test_timeline_too_few_dates():
    frame = make_frame().head(15)
    report = dm.timeline_matrix(frame, frame.copy(), "date")
    assert report["available"] is False


# ---------------- recommendations ----------------
def test_recs_stable():
    report = dm.analyze_drift(make_frame(), make_frame(), include_adversarial=False)
    recs = dm.build_recommendations(report)
    assert recs[0]["icon"] == "✅", recs[0]


def test_recs_drift():
    report = dm.analyze_drift(make_frame(), make_frame(shift=70))
    recs = dm.build_recommendations(report)
    icons = [rec["icon"] for rec in recs]
    assert "🚨" in icons, icons
    assert "🧠" in icons, icons  # AUC عالٍ => إعادة تدريب


def test_recs_watch():
    cur = make_frame()
    cur["amount"] = cur["amount"] + 4  # إزاحة صغيرة => نطاق المراقبة
    report = dm.analyze_drift(make_frame(), cur, include_adversarial=False)
    status = report["features"].set_index("feature").loc["amount", "status"]
    recs = dm.build_recommendations(report)
    if status == "watch":
        assert any(rec["icon"] == "🟠" for rec in recs), [r["icon"] for r in recs]
    else:
        assert any(rec["icon"] in ("✅", "🚨") for rec in recs)


def test_recs_missing_and_new_categories():
    cur = make_frame(missing=0.25).copy()
    cur.loc[cur.index[:120], "city"] = "Salt"
    report = dm.analyze_drift(make_frame(), cur, include_adversarial=False)
    icons = [rec["icon"] for rec in dm.build_recommendations(report)]
    assert "🆕" in icons, icons
    assert "🕳️" in icons, icons


def test_recs_trend_reliable():
    dates = pd.date_range("2025-01-01", periods=900, freq="D")
    ref = pd.DataFrame({"x": rng.normal(100, 8, 900), "day": dates.astype(str)})
    cur = pd.DataFrame({
        "x": np.r_[rng.normal(100, 8, 300), rng.normal(106, 8, 300), rng.normal(114, 8, 300)],
        "day": pd.date_range("2027-01-01", periods=900, freq="D").astype(str),
    })
    report = dm.analyze_drift(ref, cur, bins=10, include_adversarial=False)
    timeline = dm.timeline_matrix(ref, cur, "day", bins=10, freq="M")
    recs = dm.build_recommendations(report, timeline)
    assert any(rec["icon"] == "📈" for rec in recs), [r["icon"] for r in recs]


def test_recs_no_trend_on_small_windows():
    dates = pd.date_range("2026-01-01", periods=120, freq="D")
    ref = make_frame(n=120).assign(day=pd.Index(dates).astype(str)).drop(columns="date")
    cur = make_frame(n=120, dates=dates).assign(day=pd.Index(dates).astype(str)).drop(columns="date")
    report = dm.analyze_drift(ref, cur, bins=10, include_adversarial=False)
    timeline = dm.timeline_matrix(ref, cur, "day", bins=10, freq="D")
    recs = dm.build_recommendations(report, timeline)
    assert not any(rec["icon"] == "📈" for rec in recs), [r["icon"] for r in recs]
    assert any(rec["icon"] == "🧩" for rec in recs), [r["icon"] for r in recs]


def test_recs_empty_features():
    report = dm.analyze_drift(pd.DataFrame(), pd.DataFrame(), include_adversarial=False)
    recs = dm.build_recommendations(report)
    assert len(recs) == 1 and recs[0]["icon"] == "ℹ️", recs


# ---------------- exports ----------------
def test_report_csv():
    report = dm.analyze_drift(make_frame(), make_frame(shift=20), bins=10)
    text = dm.report_csv(report)
    lines = text.strip().splitlines()
    assert lines[0].startswith("feature,kind,psi,status"), lines[0]
    assert len(lines) == 1 + len(report["features"]), len(lines)
    assert "amount" in text


def test_summary_markdown():
    report = dm.analyze_drift(make_frame(), make_frame(shift=20), bins=10)
    text = dm.summary_markdown(report, "ref.csv", "cur.csv")
    for token in ("# تقرير مراقبة تغيّر البيانات", "مؤشر الاستقرار", "## التوصيات", "ref.csv", "PSI"):
        assert token in text, token
    assert text.count("\n") > 15


def test_summary_markdown_defaults():
    report = dm.analyze_drift(make_frame(), make_frame(), include_adversarial=False)
    text = dm.summary_markdown(report)
    assert "—" in text
    assert "المرجع" in text


def test_report_csv_empty():
    report = dm.analyze_drift(pd.DataFrame(), pd.DataFrame(), include_adversarial=False)
    assert dm.report_csv(report).strip() == "feature,kind,psi,status"


def main() -> int:
    """مشغّل بسيط يسمح بتشغيل الملف دون pytest."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as err:
            failed.append((name, f"ASSERT: {err}"))
        except Exception as err:
            failed.append((name, f"{type(err).__name__}: {err}"))

    print(f"PASSED: {len(tests) - len(failed)}/{len(tests)}")
    if failed:
        print("\nFAILED:")
        for name, reason in failed:
            print(f"  - {name}: {reason}")
        return 1
    print("ALL UNIT TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
