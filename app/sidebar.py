"""Sidebar: data source (upload / database), market settings, and AI settings."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import ai_report as ai
import business_analytics as ba
import data_loader as dl
import db_connector as db
from app_config import DEVELOPER


def _sync_vat_to_currency():
    code = st.session_state.get("biz_currency", "JOD")
    st.session_state["biz_vat"] = int(ba.CURRENCIES[code]["vat"] * 100)


def _safe_file_name(text: str) -> str:
    """يحوّل نصًا إلى اسم ملف آمن."""
    clean = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))
    return clean.strip("._") or "data"


def _db_display_label(engine: str, meta: dict) -> str:
    """تسمية قصيرة للاتصال بدون أي بيانات حساسة."""
    info = db.ENGINES[engine]
    database = meta.get("database") or ""
    host = meta.get("host") or ""
    if info.get("file_based"):
        return f"{info['label']} · {Path(database).name}"
    return f"{info['label']} · {host}/{database}"


def _db_connect(engine: str, params: dict) -> None:
    """يفتح الاتصال، يحدّث الجداول، ويحفظ البيانات في session_state."""
    try:
        conn = db.connect(engine, **params)
    except Exception as err:
        st.sidebar.error(
            f"⚠️ تعذّر الاتصال | Connection failed: `{type(err).__name__}: {err}`"
        )
        return

    old_conn = st.session_state.get("db_conn")
    if old_conn is not None and old_conn is not conn:
        db.close(old_conn)
    old_meta = st.session_state.get("db_meta") or {}

    st.session_state["db_conn"] = conn
    st.session_state["db_meta"] = {
        "engine": engine,
        "database": params.get("database") or "",
        "host": params.get("host") or "",
        "port": params.get("port"),
        "user": params.get("user") or "",
        "label": _db_display_label(engine, params),
    }

    same_target = (
        old_meta.get("engine") == engine
        and old_meta.get("database") == (params.get("database") or "")
        and old_meta.get("host") == (params.get("host") or "")
    )
    if not same_target:
        st.session_state.pop("db_source", None)

    try:
        st.session_state["db_tables"] = db.list_tables(conn, engine)
    except Exception as err:
        st.session_state["db_tables"] = []
        st.sidebar.warning(
            f"⚠️ تم الاتصال لكن تعذّر جلب الجداول | Connected, but listing tables failed: "
            f"`{type(err).__name__}: {err}`"
        )


def _db_disconnect(keep_source: bool = True) -> None:
    """يغلق الاتصال ويمسح حالة الجداول (ويُبقي البيانات المحمّلة إن وُجدت)."""
    db.close(st.session_state.pop("db_conn", None))
    st.session_state.pop("db_meta", None)
    st.session_state.pop("db_tables", None)
    if not keep_source:
        st.session_state.pop("db_source", None)


def database_panel():
    """لوحة ربط قواعد البيانات في الشريط الجانبي. تعيد المصدر النشط أو None."""
    engine = st.sidebar.selectbox(
        "نوع قاعدة البيانات | Database engine",
        list(db.ENGINES),
        format_func=lambda code: f"{db.ENGINES[code]['icon']} {db.ENGINES[code]['label']}",
        key="db_engine",
    )
    info = db.ENGINES[engine]

    available, note = db.driver_status(engine)
    if not available:
        st.sidebar.warning(f"⚠️ {note}")

    if info["file_based"]:
        database = st.sidebar.text_input(
            info["database_label"],
            key="db_sqlite_path",
            placeholder=info["database_placeholder"],
        )
        host, port, user, password, driver = "localhost", None, "", "", None
    else:
        host = st.sidebar.text_input("المضيف | Host", value="localhost", key="db_host")
        port = st.sidebar.number_input(
            "المنفذ | Port",
            min_value=1,
            max_value=65535,
            value=int(info["default_port"] or 1),
            step=1,
            key="db_port",
        )
        database = st.sidebar.text_input(
            info["database_label"],
            value=info["default_database"],
            key="db_name",
            placeholder=info["database_placeholder"],
        )
        user = st.sidebar.text_input("المستخدم | User", key="db_user")
        password = st.sidebar.text_input(
            "كلمة المرور | Password", type="password", key="db_password"
        )
        driver = None
        if engine == "mssql":
            drivers = db.odbc_drivers()
            if drivers:
                driver = st.sidebar.selectbox(
                    "ODBC Driver", drivers, key="db_odbc_driver"
                )
            else:
                st.sidebar.error(
                    "⚠️ لا يوجد ODBC Driver لـ SQL Server على هذا الجهاز | "
                    "No SQL Server ODBC driver installed."
                )

    col_connect, col_disconnect = st.sidebar.columns([3, 2])
    with col_connect:
        do_connect = st.button(
            "🔌 اتصال | Connect", width="stretch", key="db_connect"
        )
    with col_disconnect:
        do_disconnect = st.button(
            "✖ فصل | Disconnect", width="stretch", key="db_disconnect"
        )

    if do_connect:
        _db_connect(
            engine,
            {
                "database": database,
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "driver": driver,
            },
        )
    if do_disconnect:
        _db_disconnect()
        st.sidebar.info("تم فصل الاتصال | Disconnected")

    connected = st.session_state.get("db_conn") is not None
    meta = st.session_state.get("db_meta") or {}
    same_engine = connected and meta.get("engine") == engine

    if connected:
        st.sidebar.success(f"✅ متصل | Connected: {meta.get('label', '')}")
        if not same_engine:
            st.sidebar.caption(
                "المحرك المحدد يختلف عن الاتصال الحالي — اضغط اتصال. | "
                "Selected engine differs from the current connection."
            )
        if st.sidebar.button(
            "🔄 تحديث الجداول | Refresh tables", width="stretch", key="db_refresh"
        ):
            try:
                st.session_state["db_tables"] = db.list_tables(
                    st.session_state["db_conn"], engine
                )
            except Exception as err:
                st.sidebar.error(
                    f"⚠️ تعذّر تحديث الجداول | Could not refresh tables: "
                    f"`{type(err).__name__}: {err}`"
                )
    else:
        st.sidebar.caption(
            "أدخل بيانات الاتصال ثم اضغط **اتصال**. | Fill the connection fields, then Connect."
        )

    table = None
    if same_engine:
        tables = st.session_state.get("db_tables") or []
        if tables:
            table = st.sidebar.selectbox(
                "الجدول | Table",
                tables,
                format_func=lambda item: item["label"],
                key="db_table_pick",
            )
        else:
            st.sidebar.caption(
                "لا توجد جداول معروضة — استخدم استعلامًا مخصصًا. | "
                "No tables listed — use a custom query."
            )

    custom_sql = st.sidebar.text_area(
        "استعلام مخصص (يتجاوز الجدول) | Custom SQL (overrides table)",
        key="db_custom_sql",
        placeholder="SELECT * FROM table_name",
        height=90,
    )
    row_limit = st.sidebar.number_input(
        "حد الصفوف | Row limit",
        min_value=1,
        max_value=db.MAX_TABLE_LIMIT,
        value=db.DEFAULT_TABLE_LIMIT,
        step=100,
        key="db_row_limit",
    )

    if st.sidebar.button(
        "📥 تحميل البيانات | Load data", width="stretch", key="db_load"
    ):
        if not connected:
            st.sidebar.error("اتصل بقاعدة البيانات أولًا | Connect to the database first.")
        elif not same_engine:
            st.sidebar.error("أعد الاتصال بالمحرك المحدد | Reconnect with the selected engine.")
        else:
            sql = (custom_sql or "").strip()
            used_custom = bool(sql)
            if not sql:
                if table is None:
                    st.sidebar.error(
                        "اختر جدولًا أو اكتب استعلامًا | Choose a table or write a query."
                    )
                else:
                    sql = db.build_table_query(engine, table, int(row_limit))

            if sql:
                ok, reason = db.is_read_query(sql)
                if not ok:
                    st.sidebar.error(f"⚠️ {reason}")
                else:
                    try:
                        df = db.fetch_df(
                            st.session_state["db_conn"], engine, sql, int(row_limit)
                        )
                    except Exception as err:
                        st.sidebar.error(
                            f"⚠️ تعذّر تنفيذ الاستعلام | Query failed: "
                            f"`{type(err).__name__}: {err}`"
                        )
                    else:
                        if df.empty:
                            st.sidebar.warning(
                                "الاستعلام لم يُرجع صفوفًا | Query returned no rows."
                            )
                        else:
                            payload = df.to_csv(index=False).encode("utf-8-sig")
                            if used_custom:
                                target_name = "query"
                                target_label = "استعلام مخصص | Custom query"
                            else:
                                target_name = table["name"]
                                target_label = table["label"]
                            file_name = f"{engine}-{_safe_file_name(target_name)}.csv"
                            source = {
                                "name": file_name,
                                "payload": payload,
                                "label": f"🗄️ {meta.get('label', info['label'])} · {target_label}",
                                "id": db.source_id(engine, sql, int(row_limit), len(df)),
                                "origin": "database",
                                "detail": f"{info['label']} · {target_label}",
                            }
                            st.session_state["db_source"] = source
                            st.sidebar.success(
                                f"✅ تم تحميل {len(df):,} صف × {df.shape[1]} عمود | "
                                f"Loaded {len(df):,} rows."
                            )

    db_source = st.session_state.get("db_source")
    if db_source:
        size = len(db_source["payload"])
        size_text = f"{size / 1024:,.1f} KB" if size >= 1024 else f"{size:,} B"
        st.sidebar.caption(f"🗄️ {db_source.get('detail', '')} · {size_text}")
        if st.sidebar.button(
            "🧹 إزالة البيانات المحمّلة | Clear loaded data",
            width="stretch",
            key="db_clear_source",
        ):
            st.session_state.pop("db_source", None)
            db_source = None
    else:
        st.sidebar.caption(
            "حمّل جدولًا أو استعلامًا ليُطبَّق على جميع التحليلات. | "
            "Load a table or query to apply it to all analyses."
        )

    return db_source


def sidebar_controls():
    """مصدر بيانات واحد من الشريط الجانبي + إعدادات متقدمة مطويّة.

    المصدر (ملف مرفوع أو قاعدة بيانات) يُستخدم في جميع التحليلات:
    انحرافات، أعمال، تنبؤ، ومحادثة. لا تُعرض مفاتيح أو معرّفات خاصة.
    """
    st.sidebar.markdown(
        '<div class="badge-orange">📂 البيانات | Data</div>',
        unsafe_allow_html=True,
    )

    mode_options = ["رفع ملف | Upload", "قاعدة بيانات | Database"]
    if st.session_state.get("data_mode") not in mode_options:
        st.session_state["data_mode"] = mode_options[0]
    # تنظيف حالة وضع المسار المحلي القديم بعد إزالته
    st.session_state.pop("path_source", None)
    st.session_state.pop("path_candidates", None)

    mode = st.sidebar.radio(
        "طريقة إدخال البيانات | Data source",
        mode_options,
        horizontal=True,
        key="data_mode",
        label_visibility="collapsed",
    )

    uploaded = None
    db_source = None

    if mode.startswith("رفع"):
        uploaded = st.sidebar.file_uploader(
            "ارفع ملفك هنا (CSV / Excel / JSON) | Upload your file",
            type=dl.SUPPORTED_TYPES,
            key="main_upload",
            help=(
                "ملف واحد يُطبَّق على جميع التحليلات: كشف الانحرافات، تحليلات الأعمال، "
                "التحليل التنبؤي، والمحادثة."
            ),
        )
        if uploaded is not None:
            st.sidebar.success("✅ الملف جاهز ويُطبَّق على جميع التحليلات. | Ready — applied to all analyses.")
        else:
            st.sidebar.caption("بدون ملف: تُستخدم بيانات تجريبية جاهزة لكل تبويب.")
    else:
        db_source = database_panel()

    with st.sidebar.expander("⚙️ إعدادات متقدمة | Advanced", expanded=False):
        algo = st.selectbox(
            "خوارزمية كشف الانحرافات | Detection algorithm",
            ["Isolation Forest", "Local Outlier Factor (LOF)", "One-Class SVM"],
            key="algo_choice",
        )
        contamination = st.slider(
            "نسبة الشذوذ المتوقعة | Expected anomaly rate",
            min_value=0.001, max_value=0.20, value=0.01, step=0.001,
            format="%.3f", key="contamination",
        )
        st.markdown("##### 🌍 إعدادات السوق | Market")
        currency = st.selectbox(
            "العملة | Currency",
            list(ba.CURRENCIES.keys()),
            format_func=lambda code: f"{code} — {ba.CURRENCIES[code]['name_ar']}",
            key="biz_currency",
            on_change=_sync_vat_to_currency,
        )
        default_vat = int(ba.CURRENCIES[currency]["vat"] * 100)
        vat_rate = st.slider(
            "ضريبة المبيعات % | VAT %",
            min_value=0, max_value=30, value=default_vat, step=1,
            key="biz_vat",
        ) / 100
        target_margin = st.slider(
            "الهامش المستهدف % | Target margin %",
            min_value=0, max_value=60, value=25, step=1,
            key="biz_target_margin",
        ) / 100
        forecast_days = st.slider(
            "أفق التوقّع (أيام) | Forecast horizon (days)",
            min_value=7, max_value=60, value=14, step=1,
            key="biz_forecast",
        )
        ai_auto = st.toggle(
            "توليد تقرير الذكاء الاصطناعي تلقائيًا | Auto-generate AI report",
            value=False, key="ai_auto",
        )
        run_btn = st.button("🔄 إعادة التحليل | Re-run analysis", width="stretch")

    market = {
        "currency": currency,
        "vat_rate": vat_rate,
        "target_margin": target_margin,
        "forecast_days": forecast_days,
    }
    ai_conf = {
        "model": ai.DEFAULT_MODEL,
        "key": None,
        "api_key": None,
        "auto": ai_auto,
    }

    st.sidebar.markdown("---")
    if ai.resolve_api_key(None):
        st.sidebar.caption("🟢 مساعد الذكاء الاصطناعي جاهز")
    else:
        st.sidebar.caption("⚪ وضع التحليل المحلي (بدون مفتاح AI)")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div class="dev-badge">'
        "<span>👨‍💻 المطوّر | Developer</span>"
        f"<strong>{DEVELOPER}</strong>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div style="color:#9A9AA2; font-size:0.75rem; margin-top:6px;">'
        "نظام تحليل بيانات — مشروع تخرج<br>"
        "Data Analytics System — Graduation Project"
        "</div>",
        unsafe_allow_html=True,
    )

    return (
        uploaded, db_source, mode,
        algo, contamination, run_btn, market, ai_conf,
    )
