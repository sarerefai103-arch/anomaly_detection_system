"""Data services: cached reads, analysis, drift computation, source resolution."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

import business_analytics as ba
import data_loader as dl
import drift_monitor as dm
from app_config import DEFAULT_BUSINESS_PATH


@st.cache_data(show_spinner=False)
def _read_cached(name: str, payload: bytes):
    return dl.read_dataset(io.BytesIO(payload), name=name)


@st.cache_data(show_spinner="🔄 جاري تحليل البيانات... | Analyzing data...")
def _analyze_cached(
    name: str, payload: bytes, vat_rate: float, currency: str,
    target_margin: float, forecast_days: int,
):
    df, meta = dl.read_dataset(io.BytesIO(payload), name=name)
    result = ba.analyze(
        df, vat_rate=vat_rate, currency=currency,
        target_margin=target_margin, forecast_days=forecast_days,
    )
    return df, meta, result


@st.cache_data(show_spinner=False)
def _quality_cached(name: str, payload: bytes):
    df, _ = dl.read_dataset(io.BytesIO(payload), name=name)
    return dl.data_quality(df)


# ---- مراقبة تغيّر البيانات: تخزين مؤقت على شكل pickle للحفاظ على الأنواع ----
def _pickle_df(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_pickle(buffer)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def _drift_cached(ref_bytes: bytes, cur_bytes: bytes, bins: int, adversarial: bool):
    # bytes مُولَّدة داخليًا بواسطة _pickle_df من جلسة المستخدم نفسه — لا تُقبل من خارج التطبيق
    ref = pd.read_pickle(io.BytesIO(ref_bytes))  # noqa: S301
    cur = pd.read_pickle(io.BytesIO(cur_bytes))  # noqa: S301
    return dm.analyze_drift(
        ref, cur, bins=int(bins), include_adversarial=bool(adversarial),
    )


@st.cache_data(show_spinner=False)
def _drift_timeline_cached(ref_bytes: bytes, cur_bytes: bytes, date_col: str,
                           bins: int, freq: str):
    ref = pd.read_pickle(io.BytesIO(ref_bytes))  # noqa: S301
    cur = pd.read_pickle(io.BytesIO(cur_bytes))  # noqa: S301
    return dm.timeline_matrix(
        ref, cur, date_col, bins=int(bins), freq=None if freq == "auto" else freq,
    )


def _sample_payload(path: Path) -> tuple[str, bytes]:
    path = Path(path)
    return path.name, path.read_bytes()


def resolve_business_source(uploaded, db_source=None, mode="") -> dict:
    """مصدر البيانات الموحّد حسب وضع الإدخال النشط.

    الأولوية: ملف مرفوع ← قاعدة بيانات، وإلا بيانات تجريبية.
    """
    if mode.startswith("رفع") and uploaded is not None:
        name, payload = uploaded.name, uploaded.getvalue()
        return {
            "name": name,
            "payload": payload,
            "label": f"📁 {name}",
            "id": getattr(uploaded, "file_id", None) or f"{name}:{len(payload)}",
            "origin": "upload",
        }
    if mode.startswith("قاعدة") and db_source:
        return db_source

    if not mode:
        if uploaded is not None:
            name, payload = uploaded.name, uploaded.getvalue()
            return {
                "name": name,
                "payload": payload,
                "label": f"📁 {name}",
                "id": getattr(uploaded, "file_id", None) or f"{name}:{len(payload)}",
                "origin": "upload",
            }
        if db_source:
            return db_source

    name, payload = _sample_payload(DEFAULT_BUSINESS_PATH)
    return {
        "name": name,
        "payload": payload,
        "label": "📊 داتاسيت مبيعات تجريبي | Sample sales data",
        "id": name,
        "origin": "demo",
    }
