"""Anomaly detection: preprocessing, scaling, and model fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import business_analytics as ba


def preprocess(df: pd.DataFrame):
    df = df.copy()
    label_col = "Class" if "Class" in df.columns else None

    working = ba.coerce_numeric_columns(df)
    numeric_df = working.select_dtypes(include=[np.number]).copy()
    if label_col and label_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[label_col])

    numeric_df = numeric_df.dropna(axis=1, how="all")
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    # تحقق من صلاحية البيانات قبل تدريب النموذج
    if numeric_df.shape[1] == 0:
        raise ValueError(
            "الملف لا يحتوي على أي عمود رقمي. "
            "أعمدة النصوص فقط لا يمكن تحليلها — أضف أعمدة رقمية أو حوّل الأعمدة النصية إلى أرقام. "
            "(No numeric columns found in the uploaded file.)"
        )
    if len(numeric_df) < 2:
        raise ValueError(
            f"الملف يحتوي على {len(numeric_df)} صف فقط، والتحليل يحتاج صفين على الأقل. "
            "(At least 2 rows are required.)"
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)
    return X_scaled, numeric_df.columns.tolist(), label_col, df


def run_detection(X, algo_name, contamination):
    if algo_name == "Isolation Forest":
        model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
        model.fit(X)
        scores = -model.decision_function(X)
        preds = np.where(model.predict(X) == -1, 1, 0)

    elif algo_name.startswith("Local Outlier"):
        # LOF يحتاج عدد جيران أقل من عدد الصفوف
        n_neighbors = max(1, min(20, len(X) - 1))
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
        preds_raw = model.fit_predict(X)
        scores = -model.negative_outlier_factor_
        preds = np.where(preds_raw == -1, 1, 0)

    else:  # One-Class SVM
        model = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
        model.fit(X)
        scores = -model.decision_function(X)
        preds = np.where(model.predict(X) == -1, 1, 0)

    return scores, preds
