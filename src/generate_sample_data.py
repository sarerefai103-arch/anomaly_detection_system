"""
generate_sample_data.py
------------------------
يولّد بيانات تجريبية (Synthetic) بنفس بنية داتاسيت
Kaggle Credit Card Fraud Detection:
- Time: الوقت بالثواني من أول معاملة
- V1..V28: خصائص مشفّرة (PCA) - نولّدها كتوزيع طبيعي
- Amount: قيمة المعاملة
- Class: 0 = عادية, 1 = احتيال (نادرة جداً ~0.17%)

الهدف: تشغيل واختبار خط الأنابيب (Pipeline) قبل توفر
البيانات الحقيقية من Kaggle.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(n_normal=5000, n_fraud=20, seed=42):
    rng = np.random.default_rng(seed)

    # ---- المعاملات العادية ----
    normal_data = {
        "Time": rng.integers(0, 172800, n_normal),  # يومين تقريباً بالثواني
        "Amount": np.round(rng.gamma(shape=2.0, scale=40.0, size=n_normal), 2),
    }
    for i in range(1, 29):
        normal_data[f"V{i}"] = rng.normal(loc=0.0, scale=1.0, size=n_normal)
    normal_data["Class"] = np.zeros(n_normal, dtype=int)

    # ---- معاملات احتيالية (شاذة عمداً) ----
    fraud_data = {
        "Time": rng.integers(0, 172800, n_fraud),
        "Amount": np.round(rng.gamma(shape=5.0, scale=300.0, size=n_fraud), 2),
    }
    for i in range(1, 29):
        # قيم أبعد عن المتوسط لمحاكاة السلوك الشاذ
        fraud_data[f"V{i}"] = rng.normal(loc=rng.choice([-4, 4]), scale=2.5, size=n_fraud)
    fraud_data["Class"] = np.ones(n_fraud, dtype=int)

    df_normal = pd.DataFrame(normal_data)
    df_fraud = pd.DataFrame(fraud_data)

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # خلط الصفوف

    # ترتيب الأعمدة بنفس ترتيب داتاسيت Kaggle الأصلي
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    return df[cols]


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "creditcard_sample.csv"
    df = generate_sample_data()
    df.to_csv(out_path, index=False)
    print(f"تم إنشاء {len(df)} صف بيانات")
    print(f"عدد المعاملات العادية: {(df['Class']==0).sum()}")
    print(f"عدد المعاملات الاحتيالية: {(df['Class']==1).sum()}")
    print(f"تم الحفظ في: {out_path}")
