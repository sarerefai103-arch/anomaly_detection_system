"""
eda_preprocessing.py
---------------------
المرحلة 3: استكشاف البيانات (EDA) وتنظيفها وتجهيزها.

الخطوات:
1. تحميل البيانات الخام
2. فحص القيم الناقصة والمكررة
3. عرض إحصائيات وصفية
4. توليد رسوم بيانية (توزيع الفئات، توزيع Amount)
5. تطبيع (Scaling) الأعمدة الرقمية
6. حفظ نسخة نظيفة في data/processed/
"""

import urllib.request
from pathlib import Path

import arabic_reshaper
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from bidi.algorithm import get_display
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "creditcard_sample.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "creditcard_clean.csv"
REPORTS_DIR = BASE_DIR / "reports"
FONTS_DIR = BASE_DIR / "assets" / "fonts"

# ---- إعداد خط عربي يدعم matplotlib ----
def setup_arabic_font():
    """يحمّل خط Noto Naskh Arabic (مجاني - Google Fonts) لدعم رسم النص العربي بشكل صحيح."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    font_path = FONTS_DIR / "NotoNaskhArabic-Regular.ttf"
    if not font_path.exists():
        url = "https://raw.githubusercontent.com/google/fonts/main/ofl/notonaskharabic/NotoNaskhArabic%5Bwght%5D.ttf"
        try:
            urllib.request.urlretrieve(url, font_path)
        except Exception as e:
            print(f"تعذر تحميل الخط العربي تلقائياً: {e}")
            return None
    fm.fontManager.addfont(str(font_path))
    return fm.FontProperties(fname=str(font_path)).get_name()


def ar(text: str) -> str:
    """يعالج النص العربي (يربط الحروف ويعكس الاتجاه) عشان يظهر صح بالرسم."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def bi(ar_text: str, en_text: str) -> str:
    """نص ثنائي اللغة: عربي معالج + إنجليزي، بسطرين."""
    return f"{en_text}\n{ar(ar_text)}"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"تم تحميل البيانات: {df.shape[0]} صف, {df.shape[1]} عمود")
    return df


def inspect_data(df: pd.DataFrame):
    print("\n--- معلومات عامة ---")
    print(df.info())

    print("\n--- القيم الناقصة لكل عمود ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "لا توجد قيم ناقصة ✅")

    print("\n--- الصفوف المكررة ---")
    dup_count = df.duplicated().sum()
    print(f"عدد الصفوف المكررة: {dup_count}")

    print("\n--- توزيع الفئات (Class) ---")
    class_counts = df["Class"].value_counts()
    print(class_counts)
    fraud_pct = (class_counts.get(1, 0) / len(df)) * 100
    print(f"نسبة الاحتيال: {fraud_pct:.3f}%")

    return dup_count


def plot_class_distribution(df: pd.DataFrame, save_path: Path):
    font_name = setup_arabic_font()
    if font_name:
        plt.rcParams["font.family"] = font_name

    _fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    df["Class"].value_counts().plot(
        kind="bar", ax=axes[0], color=["#4C72B0", "#C44E52"]
    )
    axes[0].set_title(bi("توزيع الفئات: عادي مقابل احتيال", "Class Distribution: Normal vs Fraud"))
    axes[0].set_xticklabels([bi("عادي", "Normal"), bi("احتيال", "Fraud")], rotation=0)
    axes[0].set_ylabel(bi("عدد المعاملات", "Number of Transactions"))

    axes[1].hist(df[df["Class"] == 0]["Amount"], bins=40, alpha=0.6, label=bi("عادي", "Normal"))
    axes[1].hist(df[df["Class"] == 1]["Amount"], bins=40, alpha=0.6, label=bi("احتيال", "Fraud"))
    axes[1].set_title(bi("توزيع قيمة المعاملة", "Transaction Amount Distribution"))
    axes[1].set_xlabel("Amount")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"\nتم حفظ الرسم البياني: {save_path}")


def clean_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.drop_duplicates().copy()
    df_clean = df_clean.dropna()

    scaler = StandardScaler()
    df_clean["Amount_scaled"] = scaler.fit_transform(df_clean[["Amount"]])
    df_clean["Time_scaled"] = scaler.fit_transform(df_clean[["Time"]])

    print(f"\nعدد الصفوف بعد التنظيف: {len(df_clean)}")
    return df_clean


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    df = load_data(RAW_PATH)
    inspect_data(df)
    plot_class_distribution(df, REPORTS_DIR / "class_distribution.png")
    df_clean = clean_and_scale(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"\nتم حفظ البيانات النظيفة في: {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
