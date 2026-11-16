"""Plotly chart builders for every tab (shared dark + orange template)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

import drift_monitor as dm
import predictive_analytics as pa

DARK_TEMPLATE = go.layout.Template(
    layout={
        "paper_bgcolor": "#1C1C21",
        "plot_bgcolor": "#1C1C21",
        "font": {"color": "#F2F2F2"},
        "xaxis": {"gridcolor": "#2A2A30"},
        "yaxis": {"gridcolor": "#2A2A30"},
        "colorway": ["#FF7A1A", "#3D3D45", "#FFB066"],
    }
)


def plot_pca_scatter(X, preds, scores):
    n_samples, n_features = X.shape
    # PCA ثنائي الأبعاد يحتاج سجلين وعمودين رقميين على الأقل
    if min(n_samples, n_features) < 2:
        return None
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "الحالة | Status": np.where(preds == 1, "شاذ | Anomaly", "طبيعي | Normal"),
        "Score": scores,
    })
    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="الحالة | Status",
        color_discrete_map={"شاذ | Anomaly": "#FF7A1A", "طبيعي | Normal": "#4A4A52"},
        opacity=0.75,
        title="توزيع النقاط بعد تقليل الأبعاد (PCA) | Data Distribution (PCA Projection)",
    )
    fig.update_traces(marker={"size": 7, "line": {"width": 0.5, "color": "#0B0B0D"}})
    fig.update_layout(template=DARK_TEMPLATE, transition_duration=600, legend_title_text="")
    return fig


def plot_score_distribution(scores, preds):
    df = pd.DataFrame({"Score": scores, "الحالة | Status": np.where(preds == 1, "شاذ | Anomaly", "طبيعي | Normal")})
    fig = px.histogram(
        df, x="Score", color="الحالة | Status", nbins=50, opacity=0.8,
        color_discrete_map={"شاذ | Anomaly": "#FF7A1A", "طبيعي | Normal": "#4A4A52"},
        title="توزيع درجات الشذوذ | Anomaly Score Distribution",
    )
    fig.update_layout(template=DARK_TEMPLATE, bargap=0.05, transition_duration=600, legend_title_text="")
    return fig


def plot_gauge(anomaly_rate):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=anomaly_rate * 100,
        number={"suffix": "%", "font": {"color": "#FF7A1A", "size": 40}},
        gauge={
            "axis": {"range": [0, max(5, anomaly_rate * 100 * 2)], "tickcolor": "#F2F2F2"},
            "bar": {"color": "#FF7A1A"},
            "bgcolor": "#1C1C21",
            "borderwidth": 0,
            "steps": [
                {"range": [0, max(5, anomaly_rate * 100 * 2)], "color": "#2A2A30"},
            ],
        },
        title={"text": "نسبة الشذوذ المكتشفة | Detected Anomaly Rate", "font": {"color": "#9A9AA2", "size": 14}},
    ))
    fig.update_layout(template=DARK_TEMPLATE, height=280, margin={"t": 60, "b": 10, "l": 20, "r": 20})
    return fig


def plot_revenue_profit(daily):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["revenue"], name="إيراد | Revenue", mode="lines",
        line={"color": "#FF7A1A", "width": 2.5}, fill="tozeroy", fillcolor="rgba(255,122,26,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["profit"], name="ربح | Profit", mode="lines",
        line={"color": "#FFB066", "width": 2},
    ))
    fig.update_layout(
        template=DARK_TEMPLATE, height=340, hovermode="x unified",
        title="الإيراد والربح يوميًا | Daily Revenue & Profit",
        margin={"t": 60, "b": 30, "l": 20, "r": 20}, legend_title_text="",
    )
    return fig


def plot_expense_vs_revenue(monthly):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly["label"], y=monthly["revenue"], name="إيراد | Revenue", marker_color="#FF7A1A"))
    fig.add_trace(go.Bar(x=monthly["label"], y=monthly["expense"], name="مصاريف | Expenses", marker_color="#3D3D45"))
    fig.add_trace(go.Scatter(
        x=monthly["label"], y=monthly["profit"], name="ربح | Profit",
        mode="lines+markers", line={"color": "#FFB066", "width": 2.5},
    ))
    fig.update_layout(
        template=DARK_TEMPLATE, barmode="group", height=340, hovermode="x unified",
        title="المصاريف مقابل الإيراد شهريًا | Expenses vs Revenue (Monthly)",
        margin={"t": 60, "b": 30, "l": 20, "r": 20}, legend_title_text="",
    )
    return fig


def plot_category_donut(services):
    fig = px.pie(
        services, names="name", values="revenue", hole=0.55,
        title="توزيع الإيراد حسب الخدمة | Revenue by Service",
    )
    fig.update_traces(
        textposition="inside", textinfo="percent+label",
        marker={"line": {"color": "#0B0B0D", "width": 2}},
    )
    fig.update_layout(
        template=DARK_TEMPLATE, height=360, showlegend=False,
        margin={"t": 60, "b": 20, "l": 10, "r": 10},
    )
    return fig


def plot_pareto(clients):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=clients["name"], y=clients["revenue"], name="إيراد | Revenue", marker_color="#FF7A1A"))
    fig.add_trace(go.Scatter(
        x=clients["name"], y=clients["cum_share"] * 100, name="تراكمي % | Cumulative %",
        yaxis="y2", mode="lines+markers", line={"color": "#FFB066", "width": 2},
    ))
    fig.update_layout(
        template=DARK_TEMPLATE, height=360,
        title="باريتو العملاء (80/20) | Customer Pareto",
        yaxis={"title": "الإيراد | Revenue"},
        yaxis2={"title": "تراكمي % | Cumulative %", "overlaying": "y", "side": "right", "range": [0, 105]},
        margin={"t": 60, "b": 90, "l": 20, "r": 20}, legend_title_text="",
    )
    return fig


def plot_status(status):
    fig = px.pie(
        status, names="label", values="value", hole=0.55,
        color="label",
        color_discrete_map={
            "مدفوع | Paid": "#FF7A1A",
            "معلق | Pending": "#FFB066",
            "ملغي/مرتجع | Lost": "#3D3D45",
            "أخرى | Other": "#2A2A30",
        },
        title="حالات الفواتير | Invoice Status",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      marker={"line": {"color": "#0B0B0D", "width": 2}})
    fig.update_layout(
        template=DARK_TEMPLATE, height=320, showlegend=False,
        margin={"t": 60, "b": 20, "l": 10, "r": 10},
    )
    return fig


def plot_forecast(daily, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["revenue"], name="فعلي | Actual", mode="lines",
        line={"color": "#9A9AA2", "width": 1.5},
    ))
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["upper"], mode="lines",
                             line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=forecast["date"], y=forecast["lower"], mode="lines", line={"width": 0},
        fill="tonexty", fillcolor="rgba(255,122,26,0.15)", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["date"], y=forecast["forecast"], name="توقّع | Forecast",
        mode="lines+markers", line={"color": "#FF7A1A", "width": 2.5, "dash": "dash"},
    ))
    fig.update_layout(
        template=DARK_TEMPLATE, height=340, hovermode="x unified",
        title="توقّع الإيراد | Revenue Forecast",
        margin={"t": 60, "b": 30, "l": 20, "r": 20}, legend_title_text="",
    )
    return fig


def plot_weekday(weekday):
    fig = px.bar(
        weekday, x="name", y="revenue",
        title="متوسط الإيراد حسب يوم الأسبوع | Average Revenue by Weekday",
        color="revenue", color_continuous_scale=["#3D3D45", "#FF7A1A"],
    )
    fig.update_layout(
        template=DARK_TEMPLATE, height=320, coloraxis_showscale=False,
        margin={"t": 60, "b": 70, "l": 20, "r": 20}, xaxis_title="", yaxis_title="",
    )
    return fig


def plot_daily_anomalies(daily):
    normal = daily[~daily["is_anomaly"]]
    odd = daily[daily["is_anomaly"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["date"], y=normal["revenue"], mode="markers", name="طبيعي | Normal",
        marker={"color": "#4A4A52", "size": 6},
    ))
    fig.add_trace(go.Scatter(
        x=odd["date"], y=odd["revenue"], mode="markers", name="شاذ | Anomaly",
        marker={"color": "#FF7A1A", "size": 11, "symbol": "diamond", "line": {"color": "#FFB066", "width": 1}},
    ))
    fig.update_layout(
        template=DARK_TEMPLATE, height=320, hovermode="x unified",
        title="الأيام الشاذة في الإيراد | Anomalous Revenue Days",
        margin={"t": 60, "b": 30, "l": 20, "r": 20}, legend_title_text="",
    )
    return fig


def plot_predictive_forecast(daily, sales, metric: str = "revenue"):
    block = sales.get(metric, {})
    fc = pa._safe_df(block.get("df"))
    hist_col = metric if metric in daily.columns else "revenue"
    titles = {
        "revenue": "توقع الإيرادات | Revenue Forecast",
        "profit": "توقع الأرباح | Profit Forecast",
        "expense": "توقع المصاريف | Expense Forecast",
    }
    colors = {"revenue": "#FF7A1A", "profit": "#FFB066", "expense": "#FF4D4D"}
    fig = go.Figure()
    if not daily.empty and hist_col in daily.columns:
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily[hist_col], name="فعلي | Actual",
            mode="lines", line={"color": colors.get(metric, "#FF7A1A"), "width": 2},
        ))
    if not fc.empty:
        fig.add_trace(go.Scatter(
            x=fc["date"], y=fc["upper"], name="حد أعلى | Upper",
            mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fc["date"], y=fc["lower"], name="فاصل الثقة | Confidence",
            mode="lines", line={"width": 0}, fill="tonexty",
            fillcolor="rgba(242,107,15,0.18)", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fc["date"], y=fc["forecast"], name="توقع | Forecast",
            mode="lines+markers", line={"color": "#00D4FF", "width": 2.5, "dash": "dot"},
            marker={"size": 4},
        ))
    fig.update_layout(
        template=DARK_TEMPLATE, height=360, hovermode="x unified",
        title=titles.get(metric, "Forecast"),
        margin={"t": 58, "b": 30, "l": 20, "r": 20}, legend_title_text="",
    )
    return fig


def plot_demand_forecast(demand_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=demand_df["service"], y=demand_df["projected_revenue"],
        name="إيراد متوقع | Projected Revenue", marker_color="#FF7A1A",
        text=demand_df["trend_icon"], textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        x=demand_df["service"], y=demand_df["projected_orders"],
        name="طلبات متوقعة | Projected Orders", yaxis="y2",
        mode="lines+markers", line={"color": "#00D4FF", "width": 2.5},
        marker={"size": 8},
    ))
    fig.update_layout(
        template=DARK_TEMPLATE, height=370,
        title="توقع الطلب على المنتجات/الخدمات | Product Demand Forecast",
        yaxis={"title": "إيراد متوقع | Revenue"},
        yaxis2={"title": "طلبات | Orders", "overlaying": "y", "side": "right"},
        margin={"t": 58, "b": 90, "l": 20, "r": 20}, legend_title_text="",
    )
    return fig


def plot_customer_segments(cust):
    if not cust.get("available"):
        return None
    seg = cust.get("segment_counts")
    if seg is None or seg.empty:
        return None
    fig = px.pie(
        seg, names="segment", values="customers", hole=0.5,
        title="شرائح العملاء (RFM) | Customer Segments",
    )
    fig.update_traces(
        textposition="inside", textinfo="percent+label",
        marker={"line": {"color": "#0B0B0D", "width": 2}},
    )
    fig.update_layout(
        template=DARK_TEMPLATE, height=360, showlegend=False,
        margin={"t": 58, "b": 20, "l": 10, "r": 10},
    )
    return fig


def plot_backtest(chart):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart["date"], y=chart["actual"], name="الفعلي | Actual", mode="lines+markers",
        line={"color": "#FF7A1A", "width": 2.5}, marker={"size": 6},
    ))
    fig.add_trace(go.Scatter(
        x=chart["date"], y=chart["predicted"], name="المتوقع | Predicted", mode="lines+markers",
        line={"color": "#00D4FF", "width": 2.5, "dash": "dash"}, marker={"size": 6},
    ))
    fig.update_layout(
        template=DARK_TEMPLATE, height=340, hovermode="x unified",
        title="الفعلي مقابل المتوقع (فترة الاختبار) | Actual vs Predicted (Test Window)",
        margin={"t": 58, "b": 30, "l": 20, "r": 20}, legend_title_text="",
    )
    return fig


def plot_drift_scores(features: pd.DataFrame):
    """رسم أعمدة أفقية لمؤشر PSI لكل عمود مع حدود القرار."""
    df = features.dropna(subset=["psi"]).copy()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="لا توجد نتائج قابلة للعرض | Nothing to plot",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            font={"color": "#9A9AA2"},
        )
        fig.update_layout(template=DARK_TEMPLATE, height=260,
                          margin={"t": 20, "b": 20, "l": 20, "r": 20})
        return fig

    df = df.sort_values("psi", ascending=True).tail(35)
    colors = [dm.STATUS_COLORS.get(status, "#8A8A93") for status in df["status"]]
    fig = go.Figure(go.Bar(
        x=df["psi"],
        y=df["feature"],
        orientation="h",
        marker={"color": colors, "line": {"color": "rgba(255,255,255,0.18)", "width": 1}},
        text=[f"{value:.3f}" for value in df["psi"]],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>PSI = %{x:.4f}<extra></extra>",
        name="PSI",
    ))
    fig.add_vline(x=dm.PSI_STABLE, line={"color": "#FFB066", "dash": "dot", "width": 1.1})
    fig.add_vline(x=dm.PSI_DRIFT, line={"color": "#FF4D4D", "dash": "dash", "width": 1.2})
    max_psi = float(df["psi"].max())
    fig.update_layout(
        template=DARK_TEMPLATE,
        height=max(340, 24 * len(df) + 130),
        margin={"t": 34, "b": 20, "l": 10, "r": 64},
        xaxis_title="PSI",
        yaxis_title="",
        xaxis={"range": [0, max(dm.PSI_DRIFT * 1.45, max_psi * 1.18)]},
        showlegend=False,
        transition_duration=500,
    )
    fig.add_annotation(x=dm.PSI_STABLE, y=1.02, yref="paper", text="0.10",
                       showarrow=False, font={"color": "#FFB066", "size": 10})
    fig.add_annotation(x=dm.PSI_DRIFT, y=1.02, yref="paper", text="0.25",
                       showarrow=False, font={"color": "#FF4D4D", "size": 10})
    return fig


def plot_drift_distribution(ref_df: pd.DataFrame, cur_df: pd.DataFrame,
                            feature: str, kind: str):
    """مقارنة توزيع العمود بين المرجع والحالي."""
    fig = go.Figure()
    if kind == "numeric":
        ref_values = pd.to_numeric(ref_df[feature], errors="coerce").dropna()
        cur_values = pd.to_numeric(cur_df[feature], errors="coerce").dropna()
        fig.add_trace(go.Histogram(
            x=ref_values, histnorm="percent", nbinsx=30,
            name="المرجع | Reference", marker_color="#5A5A63", opacity=0.75,
        ))
        fig.add_trace(go.Histogram(
            x=cur_values, histnorm="percent", nbinsx=30,
            name="الحالي | Current", marker_color="#FF7A1A", opacity=0.7,
        ))
        fig.update_layout(barmode="overlay", xaxis_title=feature,
                          yaxis_title="النسبة % | Share %")
    else:
        ref_txt = dm.as_text(ref_df[feature])
        cur_txt = dm.as_text(cur_df[feature])
        combined = pd.concat([ref_txt, cur_txt], ignore_index=True)
        top = [str(value) for value in combined.value_counts().head(12).index]
        ref_share = (
            ref_txt[ref_txt.isin(top)].value_counts(normalize=True)
            .reindex(top, fill_value=0) * 100
        )
        cur_share = (
            cur_txt[cur_txt.isin(top)].value_counts(normalize=True)
            .reindex(top, fill_value=0) * 100
        )
        fig.add_trace(go.Bar(x=top, y=ref_share, name="المرجع | Reference",
                             marker_color="#5A5A63"))
        fig.add_trace(go.Bar(x=top, y=cur_share, name="الحالي | Current",
                             marker_color="#FF7A1A"))
        fig.update_layout(barmode="group", xaxis_title=feature,
                          yaxis_title="النسبة % | Share %", xaxis_tickangle=-20)

    fig.update_layout(
        template=DARK_TEMPLATE,
        height=390,
        margin={"t": 20, "b": 10, "l": 10, "r": 10},
        legend_title_text="",
        legend={"orientation": "h", "y": 1.1},
        transition_duration=400,
    )
    return fig


def plot_drift_timeline(timeline: dict):
    """خط زمني لمتوسط/أقصى PSI مع عدد الصفوف في كل نافذة."""
    df = timeline["timeline"]
    reliable = df["reliable"] if "reliable" in df.columns else pd.Series(True, index=df.index)
    # النوافذ الصغيرة تُرسم بنقاط مفرّغة للدلالة على أنها استرشادية
    symbols = ["circle" if bool(flag) else "circle-open" for flag in reliable]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["label"], y=df["rows"], name="صفوف | Rows",
        marker_color="rgba(255,154,69,0.16)", yaxis="y2",
        hovertemplate="%{x}<br>%{y} rows<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["label"], y=df["avg_psi"], mode="lines+markers",
        name="متوسط PSI | Avg PSI",
        line={"color": "#FF7A1A", "width": 3},
        marker={"size": 8, "symbol": symbols},
        fill="tozeroy", fillcolor="rgba(255,122,26,0.10)",
        hovertemplate="%{x}<br>Avg PSI = %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["label"], y=df["max_psi"], mode="lines+markers",
        name="أعلى PSI | Max PSI",
        line={"color": "#FF4D4D", "width": 1.6, "dash": "dot"},
        marker={"size": 6, "symbol": symbols},
        hovertemplate="%{x}<br>Max PSI = %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=dm.PSI_STABLE, line={"color": "#FFB066", "dash": "dot", "width": 1})
    fig.add_hline(y=dm.PSI_DRIFT, line={"color": "#FF4D4D", "dash": "dash", "width": 1})
    fig.update_layout(
        template=DARK_TEMPLATE,
        height=410,
        margin={"t": 40, "b": 10, "l": 10, "r": 10},
        xaxis_title=f"النافذة الزمنية | {timeline['freq_label_ar']}",
        yaxis={"title": "PSI"},
        yaxis2={"title": "صفوف | Rows", "overlaying": "y", "side": "right", "showgrid": False},
        legend={"orientation": "h", "y": 1.14},
        hovermode="x unified",
        transition_duration=400,
    )
    return fig


def plot_drift_heatmap(matrix: pd.DataFrame):
    """خريطة حرارية: PSI لكل عمود عبر النوافذ الزمنية."""
    if matrix is None or matrix.empty:
        fig = go.Figure()
        fig.add_annotation(text="لا توجد بيانات كافية | Not enough data",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template=DARK_TEMPLATE, height=280)
        return fig
    values = matrix.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=values,
        x=[str(col) for col in matrix.columns],
        y=[str(row) for row in matrix.index],
        colorscale=[
            [0.0, "#151519"], [0.2, "#3A2A1A"], [0.45, "#FF9A45"],
            [0.7, "#FF5C33"], [1.0, "#B3001B"],
        ],
        zmin=0, zmax=0.6,
        colorbar={"title": "PSI", "thickness": 12},
        text=np.round(values, 3),
        texttemplate="%{text}",
        textfont={"size": 9, "color": "#F2F2F2"},
        hovertemplate="<b>%{y}</b><br>%{x}<br>PSI = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        height=max(320, 26 * len(matrix) + 150),
        margin={"t": 20, "b": 10, "l": 10, "r": 10},
        transition_duration=400,
    )
    return fig
