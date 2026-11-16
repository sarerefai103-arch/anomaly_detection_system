"""اختبار دخاني شامل: كل التبويبات + رفع ملف + مصدر قاعدة بيانات.

    python tests/test_smoke_app.py
    pytest tests/test_smoke_app.py -q

يفشل إذا أطلق أي تشغيل استثناءً داخل Streamlit.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from streamlit.testing.v1 import AppTest

NAV_HOME = "🏠 الرئيسية | Home"
NAV_ANALYTICS = "📊 التحليلات | Analytics"
NAV_CAMPAIGN = "🚀 الحملات | Campaigns"
NAV_CHAT = "💬 اسأل بياناتك | Ask Data"
AN_TABS = [
    "🔍 كشف الانحرافات | Anomaly Detection",
    "📈 تحليلات الأعمال | Business Analytics",
    "🔮 التحليل التنبؤي | Predictive Analytics",
    "🌊 تغيّر البيانات | Data Drift",
]

failures: list[str] = []
stats = {"checks": 0}


def check(name: str, at: AppTest) -> None:
    stats["checks"] += 1
    if at.exception:
        failures.extend(f"{name}: {exc.value}" for exc in at.exception)
        print(f"FAIL {name}")
    else:
        print(f"ok   {name}")


def new_app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180)
    at.run()
    return at


def make_sqlite_source(tmpdir: Path) -> Path:
    db_file = tmpdir / "smoke_source.db"
    con = sqlite3.connect(db_file)
    con.execute("CREATE TABLE sales (date TEXT, revenue REAL, expense REAL, client TEXT)")
    con.executemany(
        "INSERT INTO sales VALUES (?, ?, ?, ?)",
        [
            ("2026-01-01", 100.0, 60.0, "A"),
            ("2026-01-02", 120.0, 70.0, "B"),
            ("2026-02-01", 90.0, 65.0, "A"),
            ("2026-02-02", 140.0, 80.0, "C"),
        ],
    )
    con.commit()
    con.close()
    return db_file


def test_all_views() -> None:
    """يفتح كل تبويب مع مصدر افتراضي، ثم مصدر مرفوع، ثم مصدر قاعدة بيانات."""
    # ---------------------------------------------------------------- 1) boot
    at = new_app()
    check("boot / home", at)
    assert len(at.tabs) == 4 + 4, [t.label for t in at.tabs]

    # ------------------------------------------------------- 2) analytics tabs
    for tab in AN_TABS:
        at.session_state["main_tabs"] = NAV_ANALYTICS
        at.session_state["analytics_tabs"] = tab
        at.run()
        check(f"analytics: {tab}", at)

    # ------------------------------------------------------ 3) other main tabs
    for tab in (NAV_CAMPAIGN, NAV_CHAT, NAV_HOME):
        at.session_state["main_tabs"] = tab
        at.run()
        check(f"main: {tab}", at)

    # ------------------------------------------------------- 4) uploaded file
    csv_bytes = (ROOT / "data" / "raw" / "sales_sample.csv").read_bytes()
    at = new_app()
    uploader = at.get("file_uploader")[0]
    uploader.upload("sales_upload.csv", csv_bytes, "text/csv").run()
    check("upload sales_sample.csv", at)
    at.session_state["main_tabs"] = NAV_ANALYTICS
    for tab in AN_TABS:
        at.session_state["analytics_tabs"] = tab
        at.run()
        check(f"uploaded -> {tab}", at)
    at.session_state["main_tabs"] = NAV_CHAT
    at.run()
    check("uploaded -> chat", at)

    # ------------------------------------------------------ 5) database source
    with tempfile.TemporaryDirectory() as tmp:
        db_file = make_sqlite_source(Path(tmp))

        import db_connector as db

        conn = db.connect("sqlite", database=str(db_file))
        tables = db.list_tables(conn, "sqlite")
        sql = db.build_table_query("sqlite", tables[0], limit=100)
        df_db = db.fetch_df(conn, "sqlite", sql, limit=100)
        db.close(conn)
        print("db tables:", tables, "| rows:", len(df_db))
        if len(df_db) != 4:
            failures.append(f"sqlite fetch_df returned {len(df_db)} rows")

        payload = df_db.to_csv(index=False).encode("utf-8-sig")
        db_source = {
            "name": "sqlite-sales.csv",
            "payload": payload,
            "label": "🗄️ SQLite · sales",
            "id": db.source_id("sqlite", sql, 100, len(df_db)),
            "origin": "database",
            "detail": "SQLite · sales",
        }

        at = new_app()
        at.session_state["data_mode"] = "قاعدة بيانات | Database"
        at.session_state["db_source"] = db_source
        at.run()
        check("database source loaded", at)
        at.session_state["main_tabs"] = NAV_ANALYTICS
        for tab in AN_TABS:
            at.session_state["analytics_tabs"] = tab
            at.run()
            check(f"database -> {tab}", at)

    # ------------------------------------------------------ 6) drift modes
    at.session_state["main_tabs"] = NAV_ANALYTICS
    at.session_state["analytics_tabs"] = AN_TABS[3]
    at.session_state["drift_mode"] = "📌 لقطة محفوظة | Saved baseline"
    at.run()
    check("drift: saved baseline (empty)", at)

    print()
    print(f"{stats['checks']} checks, {len(failures)} failures")
    for failure in failures:
        print(" -", failure)
    assert not failures, f"{len(failures)} streamlit exceptions"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        test_all_views()
    except AssertionError as err:
        print("SMOKE FAILED")
        print(err)
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
