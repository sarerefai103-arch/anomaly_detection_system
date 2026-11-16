"""
db_connector.py
---------------
طبقة ربط قواعد البيانات بنظام التحليل.

تدعم قراءة البيانات فقط (Read-only) من:
    - SQLite            (مكتبة sqlite3 المدمجة)
    - PostgreSQL        (psycopg)
    - MySQL / MariaDB   (pymysql)
    - SQL Server        (pyodbc + ODBC Driver)
    - Oracle            (oracledb)

الفكرة: نقرأ جدولًا أو استعلام SELECT إلى DataFrame، ثم نحوّله إلى CSV في الذاكرة
ليُستخدم كباقي مصادر البيانات (ملف مرفوع / جدول قاعدة بيانات) في جميع التبويبات.

لا تُخزَّن كلمة المرور ولا تُعرض في أي تسمية أو تقرير.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

# ============================================================
# تعريف المحركات
# ============================================================
ENGINES: dict[str, dict[str, Any]] = {
    "sqlite": {
        "label": "SQLite",
        "icon": "🗃️",
        "module": "sqlite3",
        "default_port": None,
        "file_based": True,
        "database_label": "مسار ملف قاعدة البيانات | Database file",
        "database_placeholder": r"D:\data\app.db",
        "default_database": "",
    },
    "postgresql": {
        "label": "PostgreSQL",
        "icon": "🐘",
        "module": "psycopg",
        "default_port": 5432,
        "file_based": False,
        "database_label": "اسم قاعدة البيانات | Database name",
        "database_placeholder": "postgres",
        "default_database": "postgres",
    },
    "mysql": {
        "label": "MySQL / MariaDB",
        "icon": "🐬",
        "module": "pymysql",
        "default_port": 3306,
        "file_based": False,
        "database_label": "اسم قاعدة البيانات | Database name",
        "database_placeholder": "sales_db",
        "default_database": "",
    },
    "mssql": {
        "label": "SQL Server",
        "icon": "🪟",
        "module": "pyodbc",
        "default_port": 1433,
        "file_based": False,
        "database_label": "اسم قاعدة البيانات | Database name",
        "database_placeholder": "master",
        "default_database": "",
    },
    "oracle": {
        "label": "Oracle",
        "icon": "🔶",
        "module": "oracledb",
        "default_port": 1521,
        "file_based": False,
        "database_label": "اسم الخدمة | Service name",
        "database_placeholder": "ORCLPDB1",
        "default_database": "",
    },
}

_ENGINE_ALIASES = {
    "sqlite3": "sqlite",
    "sqlite": "sqlite",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "pg": "postgresql",
    "mysql": "mysql",
    "mariadb": "mysql",
    "mssql": "mssql",
    "sqlserver": "mssql",
    "sql server": "mssql",
    "oracle": "oracle",
}

DEFAULT_TABLE_LIMIT = 5_000
MAX_TABLE_LIMIT = 200_000

# كلمات تمنع تنفيذ أي أمر غير قراءة
_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|"
    r"call|exec|execute|attach|detach|vacuum|reindex|copy|comment|rollback|commit)\b",
    re.IGNORECASE,
)
_READ_STARTS = ("select", "with", "show", "explain", "describe", "desc", "values", "table")


def _norm(engine: str) -> str:
    key = (engine or "").strip().lower()
    if key not in _ENGINE_ALIASES:
        raise ValueError(f"Unsupported database engine: {engine!r}")
    return _ENGINE_ALIASES[key]


# ============================================================
# توفر السائقين
# ============================================================
def driver_status(engine: str) -> tuple[bool, str]:
    """(متوفر؟، رسالة توضيحية). SQLite متوفر دائمًا."""
    engine = _norm(engine)
    module = ENGINES[engine]["module"]
    if engine == "sqlite":
        return True, "sqlite3 مدمج | built-in"
    try:
        __import__(module)
        return True, f"{module} installed"
    except Exception:
        return False, f"مكتبة {module} غير مثبّتة | {module} is not installed"


def odbc_drivers() -> list[str]:
    """قائمة سائقي ODBC المتاحين على الجهاز (لـ SQL Server)."""
    try:
        import pyodbc
    except Exception:
        return []
    try:
        drivers = list(pyodbc.drivers())
    except Exception:
        return []
    sql_server = sorted((d for d in drivers if "SQL Server" in d), reverse=True)
    return sql_server or drivers


def _default_odbc_driver() -> str | None:
    drivers = odbc_drivers()
    if not drivers:
        return None
    for name in drivers:
        if "18" in name:
            return name
    return drivers[0]


# ============================================================
# الاتصال
# ============================================================
def connect(
    engine: str,
    *,
    database: str = "",
    host: str = "localhost",
    port: int | None = None,
    user: str = "",
    password: str = "",
    driver: str | None = None,
    timeout: int = 10,
) -> Any:
    """يفتح اتصالًا بقاعدة البيانات ويعيد كائن الاتصال."""
    engine = _norm(engine)
    database = (database or "").strip().strip('"').strip("'")
    host = (host or "localhost").strip() or "localhost"
    port = int(port or ENGINES[engine]["default_port"] or 0)

    if engine == "sqlite":
        if not database:
            raise ValueError("أدخل مسار ملف SQLite | Enter the SQLite database file path")
        path = Path(database).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ملف قاعدة البيانات غير موجود | Database file not found: {path}")
        return sqlite3.connect(str(path), timeout=timeout, check_same_thread=False)

    if not database:
        raise ValueError("أدخل اسم قاعدة البيانات | Enter the database name")

    if engine == "postgresql":
        import psycopg

        return psycopg.connect(
            host=host,
            port=port,
            dbname=database,
            user=user or None,
            password=password or None,
            connect_timeout=timeout,
        )

    if engine == "mysql":
        import pymysql

        return pymysql.connect(
            host=host,
            port=port,
            database=database,
            user=user or None,
            password=password or None,
            connect_timeout=timeout,
            read_timeout=max(60, timeout),
            charset="utf8mb4",
            autocommit=True,
        )

    if engine == "mssql":
        import pyodbc

        driver_name = driver or _default_odbc_driver()
        if not driver_name:
            raise RuntimeError(
                "لا يوجد ODBC Driver لـ SQL Server | No SQL Server ODBC driver installed"
            )
        return pyodbc.connect(
            driver=driver_name,
            server=f"{host},{port}" if port else host,
            database=database,
            uid=user or None,
            pwd=password or None,
            timeout=timeout,
            TrustServerCertificate="yes",
            Encrypt="yes",
        )

    if engine == "oracle":
        import oracledb

        dsn = f"{host}:{port}/{database}" if port else f"{host}/{database}"
        return oracledb.connect(
            user=user or None,
            password=password or None,
            dsn=dsn,
            tcp_connect_timeout=timeout,
        )

    raise ValueError(f"Unsupported database engine: {engine!r}")


def close(conn: Any) -> None:
    """يغلق الاتصال بهدوء."""
    if conn is None:
        return
    with contextlib.suppress(Exception):
        conn.close()


# ============================================================
# الجداول والاستعلامات
# ============================================================
def quote_identifier(engine: str, name: str) -> str:
    """يضع علامات الاقتباس حول اسم جدول/عمود حسب المحرك."""
    engine = _norm(engine)
    text = str(name).replace("\x00", "")
    if engine == "mysql":
        return "`" + text.replace("`", "``") + "`"
    if engine == "mssql":
        return "[" + text.replace("]", "]]") + "]"
    return '"' + text.replace('"', '""') + '"'


def _qualified(engine: str, table: dict[str, Any]) -> str:
    name = quote_identifier(engine, table["name"])
    schema = table.get("schema")
    if schema and schema not in ("main", "public", "dbo"):
        return f"{quote_identifier(engine, schema)}.{name}"
    if schema == "dbo" and engine == "mssql":
        return f"[dbo].{name}"
    return name


def list_tables(
    conn: Any, engine: str, schema: str | None = None
) -> list[dict[str, Any]]:
    """يعيد قائمة الجداول/العروض: [{"schema":.., "name":.., "label":..}]."""
    engine = _norm(engine)
    cur = conn.cursor()
    rows: list[tuple[str | None, str]] = []
    try:
        if engine == "sqlite":
            cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
            rows = [(None, r[0]) for r in cur.fetchall()]
        elif engine == "postgresql":
            cur.execute(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_type IN ('BASE TABLE','VIEW') "
                "AND table_schema NOT IN ('pg_catalog','information_schema') "
                "ORDER BY table_schema, table_name"
            )
            rows = [(r[0], r[1]) for r in cur.fetchall()]
        elif engine == "mysql":
            cur.execute(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_type IN ('BASE TABLE','VIEW') "
                "AND table_schema NOT IN ('mysql','information_schema','performance_schema','sys') "
                "ORDER BY table_schema, table_name"
            )
            rows = [(r[0], r[1]) for r in cur.fetchall()]
        elif engine == "mssql":
            cur.execute(
                "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
                "WHERE TABLE_TYPE IN ('BASE TABLE','VIEW') "
                "ORDER BY TABLE_SCHEMA, TABLE_NAME"
            )
            rows = [(r[0], r[1]) for r in cur.fetchall()]
        elif engine == "oracle":
            cur.execute(
                "SELECT owner, object_name FROM all_objects "
                "WHERE object_type IN ('TABLE','VIEW') "
                "AND owner NOT IN ('SYS','SYSTEM','XDB','MDSYS','CTXSYS','DBSNMP') "
                "ORDER BY owner, object_name"
            )
            rows = [(r[0], r[1]) for r in cur.fetchall()]
    finally:
        with contextlib.suppress(Exception):
            cur.close()

    wanted = (schema or "").strip().lower()
    tables: list[dict[str, Any]] = []
    for sch, name in rows:
        if not name:
            continue
        if wanted and (sch or "").lower() != wanted:
            continue
        label = name if (not sch or sch in ("main", "public", "dbo")) else f"{sch}.{name}"
        tables.append({"schema": sch, "name": name, "label": label})
    return tables


def build_table_query(engine: str, table: dict[str, Any], limit: int = DEFAULT_TABLE_LIMIT) -> str:
    """يبني استعلام قراءة آمن لجدول مع حد صفوف.

    أسماء الجداول/المخططات تُقتبس عبر ``quote_identifier``، والحد ``limit``
    عدد صحيح — لذلك لا يوجد إدخال مستخدم خام داخل نص الاستعلام.
    """
    engine = _norm(engine)
    limit = max(1, int(limit))
    target = _qualified(engine, table)
    if engine == "oracle":
        return f"SELECT * FROM {target} WHERE ROWNUM <= {limit}"  # noqa: S608
    if engine == "mssql":
        return f"SELECT TOP ({limit}) * FROM {target}"  # noqa: S608
    return f"SELECT * FROM {target} LIMIT {limit}"  # noqa: S608


def is_read_query(sql: str) -> tuple[bool, str]:
    """يتحقق أن الاستعلام قراءة فقط (SELECT/WITH/SHOW/...)."""
    text = (sql or "").strip()
    if not text:
        return False, "الاستعلام فارغ | Empty query"

    # إزالة التعليقات في البداية
    cleaned = re.sub(r"^(?:\s*(?:--[^\n]*\n|/\*.*?\*/))+", "", text, flags=re.DOTALL).strip()
    if not cleaned:
        return False, "الاستعلام فارغ | Empty query"

    # منع أكثر من أمر واحد
    body = cleaned.removesuffix(";")
    if ";" in body:
        return False, "يُسمح بأمر واحد فقط | Only a single statement is allowed"

    first = body.split(None, 1)[0].lower().strip("(")
    if first not in _READ_STARTS:
        return False, "يُسمح باستعلامات القراءة فقط (SELECT / WITH ...) | Read-only queries only"

    if _WRITE_KEYWORDS.search(body):
        return False, "الاستعلام يحتوي على أمر تعديل بيانات | Query contains a write operation"

    return True, ""


def fetch_df(conn: Any, engine: str, sql: str, limit: int = DEFAULT_TABLE_LIMIT) -> pd.DataFrame:
    """ينفّذ استعلام قراءة ويعيد DataFrame بحد أقصى للصفوف."""
    engine = _norm(engine)
    ok, reason = is_read_query(sql)
    if not ok:
        raise ValueError(reason)

    limit = max(1, int(limit))
    cur = conn.cursor()
    try:
        cur.execute(sql)
        columns = [d[0] for d in (cur.description or [])]
        if not columns:
            raise ValueError("الاستعلام لا يعيد أعمدة | Query returned no columns")
        rows = cur.fetchmany(limit)
    finally:
        with contextlib.suppress(Exception):
            cur.close()

    return pd.DataFrame.from_records(list(rows), columns=columns)


def source_id(engine: str, sql: str, limit: int, rows: int) -> str:
    """معرّف ثابت نسبيًا لمصدر قاعدة البيانات (بدون بيانات حساسة)."""
    digest = hashlib.sha1(sql.encode("utf-8", "ignore"), usedforsecurity=False).hexdigest()[:12]
    return f"db:{_norm(engine)}:{digest}:{limit}:{rows}"


def query_preview(sql: str, max_chars: int = 120) -> str:
    """نسخة مختصرة من الاستعلام لعرضها في التسميات."""
    text = " ".join((sql or "").split())
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
