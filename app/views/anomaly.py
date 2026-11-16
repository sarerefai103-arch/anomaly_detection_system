"""Anomaly detection tab."""
from __future__ import annotations

import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

from app_config import DEFAULT_DATA_PATH
from charts import plot_gauge, plot_pca_scatter, plot_score_distribution
from detection import preprocess, run_detection
from services import _read_cached, _sample_payload
from ui import kpi_card


def render_anomaly_tab(source, algo, contamination, run_btn):
    # تحميل البيانات من المصدر الموحّد في الشريط الجانبي (ملف مرفوع أو قاعدة بيانات)
    if source.get("origin") == "demo":
        name, payload = _sample_payload(DEFAULT_DATA_PATH)
        source_label = "📁 بيانات تجريبية افتراضية | Default Sample Data"
        source_id = f"demo:{name}"
    else:
        name, payload = source["name"], source["payload"]
        source_label = source["label"]
        source_id = source["id"]

    try:
        df_raw, _meta = _read_cached(name, payload)
    except Exception as err:
        st.error(f"⚠️ تعذر قراءة الملف | Could not read the file: `{type(err).__name__}: {err}`")
        return

    st.markdown(
        f'<span class="badge-orange">{source_label} — {len(df_raw)} صف / rows</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    with st.expander("👁️ معاينة البيانات الخام | Raw Data Preview"):
        st.dataframe(df_raw.head(20), width="stretch")

    # التشغيل تلقائي عند تغيير الملف/الخوارزمية، مع زر إعادة يدوي في الإعدادات المتقدمة
    signature = (source_id, algo, float(contamination))
    cached = st.session_state.get("anomaly_cache") or {}
    needs_run = (
        run_btn
        or "results" not in cached
        or cached.get("signature") != signature
    )

    if needs_run:
        with st.spinner("🔄 جاري تحليل البيانات..."):
            try:
                X, _feature_names, label_col, df_full = preprocess(df_raw)
                scores, preds = run_detection(X, algo, contamination)
            except ValueError as err:
                st.session_state.pop("anomaly_cache", None)
                st.error(f"⚠️ تعذر تحليل البيانات | Could not analyze the data\n\n{err}")
                return
            except Exception as err:
                st.session_state.pop("anomaly_cache", None)
                st.error(f"⚠️ حدث خطأ غير متوقع | Unexpected error: `{type(err).__name__}: {err}`")
                return

            st.session_state["anomaly_cache"] = {
                "signature": signature,
                "results": {
                    "X": X, "scores": scores, "preds": preds,
                    "label_col": label_col, "df_full": df_full, "algo": algo,
                },
            }

    res = st.session_state["anomaly_cache"]["results"]
    X, scores, preds = res["X"], res["scores"], res["preds"]
    label_col, df_full, algo = res["label_col"], res["df_full"], res["algo"]

    n_anomalies = int(preds.sum())
    anomaly_rate = n_anomalies / len(preds)

    # ---------------- KPI Cards ----------------
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, f"{len(df_full):,}", "إجمالي السجلات", "Total Records", 0.0)
    kpi_card(c2, f"{n_anomalies:,}", "الانحرافات المكتشفة", "Anomalies Detected", 0.1)
    kpi_card(c3, f"{anomaly_rate*100:.2f}%", "نسبة الشذوذ", "Anomaly Rate", 0.2)
    kpi_card(c4, algo, "الخوارزمية المستخدمة", "Algorithm Used", 0.3)

    st.write("")

    # ---------------- تقييم (إذا فيه Class حقيقي) ----------------
    if label_col:
        y_true = df_full[label_col].values
        try:
            roc = roc_auc_score(y_true, scores)
            pr = average_precision_score(y_true, scores)
            e1, e2 = st.columns(2)
            kpi_card(e1, f"{roc:.3f}", "ROC-AUC", "ROC-AUC", 0.35)
            kpi_card(e2, f"{pr:.3f}", "PR-AUC", "PR-AUC", 0.4)
        except Exception:
            pass

    st.write("")

    # ---------------- الرسوم البيانية ----------------
    g1, g2 = st.columns([2, 1])
    with g1, st.container(border=True, key="card_pca"):
        pca_fig = plot_pca_scatter(X, preds, scores)
        if pca_fig is None:
            st.info(
                "لا يمكن رسم PCA: يلزم عمودان رقميان وسجلان على الأقل. "
                "| PCA needs at least 2 numeric columns and 2 rows."
            )
        else:
            st.plotly_chart(pca_fig, width="stretch")
    with g2, st.container(border=True, key="card_gauge"):
        st.plotly_chart(plot_gauge(anomaly_rate), width="stretch")

    with st.container(border=True, key="card_hist"):
        st.plotly_chart(plot_score_distribution(scores, preds), width="stretch")

    # ---------------- جدول أعلى الحالات الشاذة ----------------
    with st.container(border=True, key="card_table"):
        st.markdown('<div class="section-title">🔎 أعلى الحالات الشاذة | Top Anomalies</div>', unsafe_allow_html=True)

        result_df = df_full.copy()
        result_df["Anomaly_Score"] = scores
        result_df["Is_Anomaly"] = preds
        top_anomalies = result_df[result_df["Is_Anomaly"] == 1].sort_values("Anomaly_Score", ascending=False)

        st.dataframe(top_anomalies.head(50), width="stretch")

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ تحميل النتائج الكاملة CSV | Download Full Results",
            data=csv_bytes,
            file_name="anomaly_detection_results.csv",
            mime="text/csv",
            width="content",
        )
