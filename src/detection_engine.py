"""
detection_engine.py
---------------------
المرحلة 4: محرك كشف الانحرافات (Anomaly Detection Engine)

يطبّق أكثر من خوارزمية على البيانات النظيفة، ويقيّم أداء كل
واحدة بمقاييس مناسبة للبيانات غير المتوازنة (Imbalanced Data):
Precision, Recall, F1-Score, ROC-AUC, PR-AUC

الخوارزميات:
1. Isolation Forest  (sklearn)
2. Local Outlier Factor (sklearn)
3. Autoencoder (اختياري - PyTorch/Keras) -- يُضاف بمرحلة لاحقة
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor

# نعيد استخدام دوال دعم العربي من سكربت المرحلة 3
from eda_preprocessing import bi, setup_arabic_font

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "creditcard_clean.csv"
REPORTS_DIR = BASE_DIR / "reports"

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
# نسبة الشذوذ المتوقعة (contamination) نحسبها من البيانات الحقيقية عادة،
# هون منعرفها لأنه عنا Class (لأغراض التقييم فقط - النموذج نفسه ما بشوفها بالتدريب)
CONTAMINATION = 0.01  # تقدير عملي؛ عادة أعلى شوي من النسبة الحقيقية لتفادي الفوت


def load_processed_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH)
    print(f"تم تحميل {len(df)} صف للتدريب/الاختبار")
    return df


def run_isolation_forest(X: pd.DataFrame) -> np.ndarray:
    """
    يرجّع scores حيث القيم الأعلى = أكثر شذوذاً (بعد التحويل)
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    # decision_function: قيم أعلى = طبيعي أكثر. نعكسها عشان الأعلى = شاذ أكثر
    raw_scores = model.decision_function(X)
    anomaly_scores = -raw_scores
    predictions = model.predict(X)  # -1 = شاذ, 1 = طبيعي
    predictions = np.where(predictions == -1, 1, 0)  # نحوّلها لصيغة Class (1=fraud)
    return anomaly_scores, predictions


def run_lof(X: pd.DataFrame) -> np.ndarray:
    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=CONTAMINATION,
        n_jobs=-1,
    )
    predictions = model.fit_predict(X)  # -1 = شاذ, 1 = طبيعي
    anomaly_scores = -model.negative_outlier_factor_
    predictions = np.where(predictions == -1, 1, 0)
    return anomaly_scores, predictions


def evaluate_model(y_true, y_pred, scores, model_name: str) -> dict:
    print(f"\n{'='*50}")
    print(f"تقييم النموذج: {model_name}")
    print(f"{'='*50}")

    report = classification_report(
        y_true, y_pred, target_names=["Normal", "Fraud"], zero_division=0
    )
    print(report)

    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "scores": scores,
    }


def plot_comparison(results: list, y_true, save_path: Path):
    font_name = setup_arabic_font()
    if font_name:
        plt.rcParams["font.family"] = font_name

    _fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- مقارنة AUC ---
    names = [r["model"] for r in results]
    roc_scores = [r["roc_auc"] for r in results]
    pr_scores = [r["pr_auc"] for r in results]

    x = np.arange(len(names))
    width = 0.35
    axes[0].bar(x - width/2, roc_scores, width, label="ROC-AUC", color="#4C72B0")
    axes[0].bar(x + width/2, pr_scores, width, label="PR-AUC", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylim(0, 1)
    axes[0].set_title(bi("مقارنة أداء الخوارزميات", "Algorithm Performance Comparison"))
    axes[0].legend()

    # --- Precision-Recall curve ---
    for r in results:
        precision, recall, _ = precision_recall_curve(y_true, r["scores"])
        axes[1].plot(recall, precision, label=r["model"])
    axes[1].set_xlabel(bi("الاستدعاء", "Recall"))
    axes[1].set_ylabel(bi("الدقة", "Precision"))
    axes[1].set_title(bi("منحنى Precision-Recall", "Precision-Recall Curve"))
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"\nتم حفظ رسم المقارنة: {save_path}")


def main():
    df = load_processed_data()
    X = df[FEATURE_COLS]
    y_true = df["Class"].values

    results = []

    scores_if, preds_if = run_isolation_forest(X)
    results.append(evaluate_model(y_true, preds_if, scores_if, "Isolation Forest"))

    scores_lof, preds_lof = run_lof(X)
    results.append(evaluate_model(y_true, preds_lof, scores_lof, "LOF"))

    plot_comparison(results, y_true, REPORTS_DIR / "model_comparison.png")

    # حفظ ملخص جدولي بسيط
    summary = pd.DataFrame([
        {"Model": r["model"], "ROC_AUC": r["roc_auc"], "PR_AUC": r["pr_auc"]}
        for r in results
    ])
    summary_path = REPORTS_DIR / "model_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nتم حفظ ملخص المقارنة: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
