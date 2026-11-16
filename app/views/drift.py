"""Data drift tab: reference selection, metrics, timeline, and export."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

import data_loader as dl
import drift_monitor as dm
from app_config import DRIFT_MODE_FILE, DRIFT_MODE_SNAP, DRIFT_MODE_TIME
from charts import (
    plot_drift_distribution,
    plot_drift_heatmap,
    plot_drift_scores,
    plot_drift_timeline,
)
from services import (
    _drift_cached,
    _drift_timeline_cached,
    _pickle_df,
    _read_cached,
)
from ui import group_label, kpi_card


def _drift_snapshot_save(df: pd.DataFrame, label: str) -> None:
    st.session_state["drift_snapshot"] = {
        "payload": _pickle_df(df),
        "label": label,
        "rows": len(df),
        "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }


def _drift_time_split(df: pd.DataFrame, date_cols: list[str]) -> dict | None:
    """تقسيم زمني: أول X% مرجع، والباقي هو الفترة الحالية."""
    if not date_cols:
        st.warning(
            "لا يوجد عمود تاريخ صالح للتقسيم الزمني. استخدم ملفًا مرجعيًا أو لقطة محفوظة. "
            "| No valid date column — use a reference file or a saved baseline."
        )
        return None

    c1, c2 = st.columns([1.15, 1])
    date_col = c1.selectbox("عمود التاريخ | Date column", date_cols, key="drift_date_col")
    ratio = c2.slider(
        "نسبة المرجع | Reference share", 50, 90, 70, 5,
        format="%d%%", key="drift_ratio",
    )

    dates = dm.parse_dates(df[date_col])
    valid = dates.notna()
    if int(valid.sum()) < 40:
        st.warning(
            "عدد التواريخ الصالحة غير كافٍ لبناء تقسيم زمني موثوق (يلزم 40 على الأقل). "
            "| Not enough valid dates for a reliable split."
        )
        return None

    ordered = df.loc[valid].copy()
    ordered["__drift_sort"] = dates[valid].to_numpy()
    ordered = ordered.sort_values("__drift_sort")
    split = int(len(ordered) * int(ratio) / 100)
    if split < 20 or (len(ordered) - split) < 20:
        st.warning(
            "التقسيم يترك أقل من 20 صفًا في إحدى المجموعتين — عدّل النسبة. "
            "| The split leaves fewer than 20 rows on one side."
        )
        return None

    ref_df = ordered.iloc[:split].drop(columns="__drift_sort")
    cur_df = ordered.iloc[split:].drop(columns="__drift_sort")

    start = pd.Timestamp(ordered["__drift_sort"].iloc[0]).date()
    ref_end = pd.Timestamp(ordered["__drift_sort"].iloc[split - 1]).date()
    cur_start = pd.Timestamp(ordered["__drift_sort"].iloc[split]).date()
    end = pd.Timestamp(ordered["__drift_sort"].iloc[-1]).date()
    st.caption(
        f"المرجع: {len(ref_df):,} صف ({start} → {ref_end}) · "
        f"الحالي: {len(cur_df):,} صف ({cur_start} → {end})"
    )
    return {
        "ref": ref_df,
        "cur": cur_df,
        "ref_label": f"🕒 مرجع زمني | Time baseline ({date_col})",
        "cur_label": f"🕒 الفترة الحالية | Current window ({cur_start} → {end})",
    }


def render_drift_tab(source):
    group_label("مراقبة تغيّر البيانات", "Data Drift Monitoring", "🌊")
    st.caption(
        "قارن بياناتك الحالية بمجموعة مرجعية (Baseline) لكشف تغيّر التوزيعات قبل أن يؤثر على النماذج والقرارات. "
        "| Compare current data against a baseline to catch distribution shifts early."
    )

    try:
        df_cur, _meta = _read_cached(source["name"], source["payload"])
    except Exception as err:
        st.error(f"⚠️ تعذر قراءة البيانات | Could not read data: `{type(err).__name__}: {err}`")
        return
    if df_cur.empty:
        st.warning("المصدر فارغ | The data source is empty.")
        return

    date_cols_cur = dm.detect_date_columns(df_cur)
    ref_df = cur_df = None
    ref_label = "—"
    cur_label = f"{source['label']} · {len(df_cur):,} صف"

    with st.container(border=True, key="card_drift_base"):
        st.markdown(
            '<div class="section-title">🧭 المرجع والبيانات الحالية | Baseline & Current</div>',
            unsafe_allow_html=True,
        )
        mode = st.radio(
            "طريقة تحديد المرجع | Baseline method",
            [DRIFT_MODE_TIME, DRIFT_MODE_FILE, DRIFT_MODE_SNAP],
            horizontal=True,
            key="drift_mode",
        )

        left, right = st.columns([1.5, 1])
        with left:
            if mode == DRIFT_MODE_TIME:
                split = _drift_time_split(df_cur, date_cols_cur)
                if split:
                    ref_df, cur_df = split["ref"], split["cur"]
                    ref_label, cur_label = split["ref_label"], split["cur_label"]

            elif mode == DRIFT_MODE_FILE:
                uploaded_ref = st.file_uploader(
                    "ارفع ملف المرجع | Upload reference file",
                    type=["csv", "xlsx", "xls", "json"],
                    key="drift_ref_upload",
                )
                if uploaded_ref is None:
                    st.info(
                        "ارفع ملفًا مرجعيًا (نفس بنية البيانات) للمقارنة. "
                        "| Upload a baseline file with the same columns."
                    )
                else:
                    try:
                        candidate, _ = dl.read_dataset(uploaded_ref, name=uploaded_ref.name)
                    except Exception as err:
                        st.error(f"⚠️ تعذر قراءة الملف المرجعي | Could not read reference: `{err}`")
                    else:
                        common = [c for c in candidate.columns if c in df_cur.columns]
                        extra = [c for c in candidate.columns if c not in df_cur.columns]
                        if candidate.empty:
                            st.warning("الملف المرجعي فارغ | The reference file is empty.")
                        elif not common:
                            st.error(
                                "لا توجد أعمدة مشتركة بين المرجع والبيانات الحالية. "
                                "| No shared columns between reference and current data."
                            )
                        else:
                            ref_df, cur_df = candidate, df_cur
                            ref_label = f"📁 {uploaded_ref.name} · {len(candidate):,} صف"
                            note = f"أعمدة مشتركة: {len(common)} / {candidate.shape[1]}"
                            if extra:
                                note += f" · غير موجودة بالحالي: {', '.join(map(str, extra[:4]))}"
                            st.caption(note)

            else:  # DRIFT_MODE_SNAP
                snapshot = st.session_state.get("drift_snapshot")
                if not snapshot:
                    st.info(
                        "لا توجد لقطة محفوظة بعد. احفظ البيانات الحالية كمرجع، ثم قارن بها أي مصدر لاحقًا. "
                        "| No saved baseline yet — save the current data first."
                    )
                    if st.button(
                        "📌 احفظ البيانات الحالية كمرجع | Save current as baseline",
                        key="drift_snap_save", width="stretch",
                    ):
                        _drift_snapshot_save(df_cur, source["label"])
                        st.rerun()
                else:
                    st.markdown(
                        f'<span class="badge-orange">📌 {snapshot["label"]} · '
                        f'{snapshot["rows"]:,} صف · {snapshot["time"]}</span>',
                        unsafe_allow_html=True,
                    )
                    b1, b2 = st.columns(2)
                    if b1.button("🔄 تحديث اللقطة | Refresh", key="drift_snap_update",
                                 width="stretch"):
                        _drift_snapshot_save(df_cur, source["label"])
                        st.rerun()
                    if b2.button("🗑️ مسح اللقطة | Clear", key="drift_snap_clear",
                                 width="stretch"):
                        st.session_state.pop("drift_snapshot", None)
                        st.rerun()
                    ref_df = pd.read_pickle(io.BytesIO(snapshot["payload"]))  # noqa: S301
                    cur_df = df_cur
                    ref_label = f"📌 لقطة محفوظة | Saved baseline ({snapshot['time']})"

        with right:
            bins = st.slider(
                "عدد الفئات لكل عمود | Bins per feature", 5, 20, 10, 1,
                key="drift_bins",
            )
            adversarial = st.toggle(
                "فحص الفصل الذكي | Adversarial validation (AUC)",
                value=True, key="drift_adv",
            )
            st.caption(
                f"PSI < {dm.PSI_STABLE:.2f} مستقر · "
                f"{dm.PSI_STABLE:.2f}–{dm.PSI_DRIFT:.2f} مراقبة · "
                f"> {dm.PSI_DRIFT:.2f} انحراف."
            )

    if ref_df is None or cur_df is None:
        return
    if len(ref_df) < dm.MIN_ROWS or len(cur_df) < dm.MIN_ROWS:
        st.warning(
            f"يلزم {dm.MIN_ROWS} صفًا على الأقل في كل مجموعة. "
            "| At least 20 rows are required on each side."
        )
        return

    with st.spinner("⏳ جاري حساب مؤشرات تغيّر البيانات... | Computing drift metrics..."):
        try:
            report = _drift_cached(
                _pickle_df(ref_df), _pickle_df(cur_df), int(bins), bool(adversarial),
            )
        except Exception as err:
            st.error(f"⚠️ تعذّر حساب الانحراف | Drift analysis failed: `{type(err).__name__}: {err}`")
            return

    features = report["features"]
    if features.empty:
        st.warning(
            "لا توجد أعمدة قابلة للتحليل (تواريخ/معرّفات فقط أو بيانات غير كافية). "
            "| No analyzable columns."
        )
        return

    summary = report["summary"]
    level_cls = {"stable": "badge-green", "watch": "badge-orange", "drift": "badge-red"}[summary["level"]]
    level_icon = {"stable": "✅", "watch": "🟠", "drift": "🚨"}[summary["level"]]
    st.markdown(
        f'<span class="{level_cls}">{level_icon} {summary["label_ar"]} | '
        f'{summary["label_en"]}</span>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, f"{summary['stability']:.0f}/100", "استقرار البيانات", "Stability Score", 0.0)
    kpi_card(k2, summary["drift_count"], "أعمدة منحرفة", "Drifted Features", 0.05)
    kpi_card(k3, f"{summary['max_psi']:.2f}", "أعلى مؤشر PSI", "Max PSI", 0.10)
    adversarial_result = report["adversarial"]
    auc_text = f"{adversarial_result['auc']:.2f}" if adversarial_result.get("available") else "—"
    kpi_card(k4, auc_text, "فصل المرجع عن الحالي", "Adversarial AUC", 0.15)

    st.caption(
        f"المرجع: {ref_label} · {len(ref_df):,} صف ← الحالي: {cur_label} · {len(cur_df):,} صف. "
        f"أعمدة محلّلة: {summary['feature_count']} "
        f"(مستقر {summary['stable_count']} · مراقبة {summary['watch_count']} · "
        f"انحراف {summary['drift_count']}). أعلى انحراف: {summary['max_feature']}."
    )
    if adversarial_result.get("available"):
        st.caption(f"🧠 {adversarial_result['note_ar']} (AUC = {adversarial_result['auc']:.2f})")

    if report["skipped"]:
        with st.expander(f"ℹ️ أعمدة مستبعدة من التحليل ({len(report['skipped'])})"):
            skipped_df = pd.DataFrame(report["skipped"]).rename(columns={
                "feature": "العمود | Feature",
                "reason_ar": "السبب | Reason",
                "reason_en": "Reason (EN)",
            })
            st.dataframe(
                skipped_df[["العمود | Feature", "السبب | Reason"]],
                hide_index=True, width="stretch",
            )

    group_label("مؤشر الانحراف لكل عمود", "Drift Score by Feature", "🌡️")
    with st.container(border=True, key="card_drift_scores"):
        st.plotly_chart(plot_drift_scores(features), width="stretch")

    group_label("مقارنة التوزيعات", "Distribution Comparison", "🔬")
    feature_options = features["feature"].tolist()
    if st.session_state.get("drift_feature") not in feature_options:
        st.session_state["drift_feature"] = feature_options[0]

    d1, d2 = st.columns([1, 1.2])
    selected_feature = d1.selectbox(
        "العمود | Feature", feature_options, key="drift_feature",
    )
    selected_row = features.set_index("feature").loc[selected_feature]
    d2.caption(
        f"النوع: {'رقمي | Numeric' if selected_row['kind'] == 'numeric' else 'فئوي | Categorical'} · "
        f"PSI = {selected_row['psi']:.3f} · الحالة: {dm.STATUS_AR[selected_row['status']]}"
    )
    with st.container(border=True, key="card_drift_dist"):
        st.plotly_chart(
            plot_drift_distribution(ref_df, cur_df, selected_feature, selected_row["kind"]),
            width="stretch",
        )

    group_label("تطوّر الانحراف عبر الزمن", "Drift Over Time", "⏱️")
    timeline_result = None
    timeline_cols = dm.detect_date_columns(cur_df)
    if not timeline_cols:
        st.info(
            "لا يوجد عمود تاريخ صالح في البيانات الحالية — أضف عمود تاريخ لعرض تطوّر الانحراف. "
            "| No valid date column for a timeline."
        )
    else:
        if st.session_state.get("drift_time_col") not in timeline_cols:
            st.session_state["drift_time_col"] = timeline_cols[0]
        freq_options = ["تلقائي | Auto", "يومي | Daily", "أسبوعي | Weekly", "شهري | Monthly"]
        freq_map = {
            "تلقائي | Auto": "auto", "يومي | Daily": "D",
            "أسبوعي | Weekly": "W", "شهري | Monthly": "M",
        }
        if st.session_state.get("drift_time_freq") not in freq_options:
            st.session_state["drift_time_freq"] = freq_options[0]

        t1, t2 = st.columns([1.2, 1])
        time_col = t1.selectbox("عمود الزمن | Date column", timeline_cols, key="drift_time_col")
        freq_label = t2.selectbox("التجميع | Granularity", freq_options, key="drift_time_freq")

        timeline_result = _drift_timeline_cached(
            _pickle_df(ref_df), _pickle_df(cur_df), time_col, int(bins), freq_map[freq_label],
        )
        if not timeline_result.get("available"):
            st.info(f"ℹ️ {timeline_result.get('note_ar', '')}")
        else:
            with st.container(border=True, key="card_drift_timeline"):
                st.plotly_chart(plot_drift_timeline(timeline_result), width="stretch")
                st.caption(
                    f"{timeline_result['note_ar']} "
                    f"التجميع: {timeline_result['freq_label_ar']}."
                    + (
                        " النقاط المفرّغة = نوافذ صغيرة استرشادية | Hollow markers = indicative only."
                        if timeline_result.get("unreliable_windows") else ""
                    )
                )
            with st.expander("🔥 خريطة حرارية: PSI لكل عمود عبر الزمن | Drift heatmap"):
                st.plotly_chart(plot_drift_heatmap(timeline_result["matrix"]), width="stretch")

    group_label("جدول الانحراف التفصيلي", "Detailed Drift Table", "🧾")
    table = pd.DataFrame({
        "العمود | Feature": features["feature"],
        "النوع | Type": features["kind"].map({
            "numeric": "رقمي | Numeric", "categorical": "فئوي | Categorical",
        }),
        "PSI": features["psi"],
        "الحالة | Status": features["status"].map(
            lambda status: f"{dm.STATUS_AR[status]} | {dm.STATUS_EN[status]}"
        ),
        "الاختبار | Test": features["test"],
        "p-value": features["p_value"],
        "تغيّر المتوسط % | Mean Δ": features["mean_shift_pct"],
        "فرق النواقص % | Missing Δ": features["missing_delta"] * 100.0,
        "فئات جديدة % | New cats": features["new_categories_pct"] * 100.0,
        "قيم مميزة (مرجع/حالي) | Unique": (
            features["ref_unique"].astype(str) + " / " + features["cur_unique"].astype(str)
        ),
    })
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=min(520, 38 * len(table) + 42),
        column_config={
            "PSI": st.column_config.NumberColumn(format="%.3f"),
            "p-value": st.column_config.NumberColumn(format="%.4f"),
            "تغيّر المتوسط % | Mean Δ": st.column_config.NumberColumn(format="%+.1f"),
            "فرق النواقص % | Missing Δ": st.column_config.NumberColumn(format="%+.1f"),
            "فئات جديدة % | New cats": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "⬇️ تنزيل تقرير CSV | Download CSV",
        data=dm.report_csv(report).encode("utf-8-sig"),
        file_name="data_drift_report.csv",
        mime="text/csv",
        key="drift_download_csv",
        width="stretch",
    )
    dl2.download_button(
        "⬇️ تنزيل ملخص Markdown | Download Markdown",
        data=dm.summary_markdown(report, ref_label, cur_label).encode("utf-8"),
        file_name="data_drift_report.md",
        mime="text/markdown",
        key="drift_download_md",
        width="stretch",
    )

    group_label("التوصيات", "Recommendations", "🧠")
    for rec in dm.build_recommendations(report, timeline_result):
        st.markdown(
            f'<div class="drift-rec">'
            f'<div class="drift-rec-title">{rec["icon"]} {rec["title_ar"]} '
            f'<em>| {rec["title_en"]}</em></div>'
            f'<div class="drift-rec-body">{rec["body_ar"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
