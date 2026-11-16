"""Shared configuration: paths, navigation labels, tabs, and drift modes."""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent


DEFAULT_DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard_sample.csv"
DEFAULT_BUSINESS_PATH = BASE_DIR / "data" / "raw" / "sales_sample.csv"
CSS_PATH = Path(__file__).resolve().parent / "assets" / "theme.css"
DEVELOPER = "sari refai"

# تسميات التنقل الرئيسي (تُستخدم أيضًا للتبديل البرمجي من الرئيسية)
NAV_HOME = "🏠 الرئيسية | Home"
NAV_ANALYTICS = "📊 التحليلات | Analytics"
NAV_CAMPAIGN = "🚀 الحملات | Campaigns"
NAV_CHAT = "💬 اسأل بياناتك | Ask Data"

AN_TAB_ANOMALY = "🔍 كشف الانحرافات | Anomaly Detection"
AN_TAB_BUSINESS = "📈 تحليلات الأعمال | Business Analytics"
AN_TAB_PREDICTIVE = "🔮 التحليل التنبؤي | Predictive Analytics"
AN_TAB_DRIFT = "🌊 تغيّر البيانات | Data Drift"

DRIFT_MODE_TIME = "🕒 تقسيم زمني | Auto time split"
DRIFT_MODE_FILE = "📁 ملف مرجعي | Reference file"
DRIFT_MODE_SNAP = "📌 لقطة محفوظة | Saved baseline"
