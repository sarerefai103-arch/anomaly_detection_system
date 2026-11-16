"""Ask-your-data chat tab."""
from __future__ import annotations

import streamlit as st

import ai_report as ai
import chat_with_data as chat
from services import _analyze_cached
from ui import group_label


def render_chat_tab(market, ai_conf, source):
    group_label("اسأل بياناتك", "Ask Your Data", "💬")

    name, payload, source_label = source["name"], source["payload"], source["label"]

    try:
        _df, _meta, result = _analyze_cached(
            name, payload,
            market["vat_rate"], market["currency"],
            market["target_margin"], market["forecast_days"],
        )
    except Exception as err:
        st.error(f"⚠️ تعذر قراءة/تحليل الملف | Could not read or analyze: `{type(err).__name__}: {err}`")
        return

    st.session_state["biz_shared_df"] = _df
    st.session_state["biz_shared_source"] = source_label

    ctx = chat.build_context(result, market)
    sig = chat.signature(ctx)
    state = st.session_state
    if state.get("chat_sig") != sig:
        state["chat_sig"] = sig
        state["chat_history"] = []
    history = state.setdefault("chat_history", [])

    api_key = ai.resolve_api_key(ai_conf.get("api_key"))
    model = ai_conf.get("model", ai.DEFAULT_MODEL)
    cur = market.get("currency", "JOD")

    with st.container(border=True, key="card_chat_panel"):
        st.markdown(
            '<div class="section-title">💬 مساعد البيانات | Data Assistant</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"إجمالي الإيراد {chat.money(ctx['totals']['revenue'], cur)} · "
            f"صافي الربح {chat.money(ctx['totals']['profit'], cur)} · "
            f"الهامش {chat.pct(ctx['totals']['margin_pct'])}"
        )

        pending = None
        sug_cols = st.columns(3)
        for index, suggestion in enumerate(chat.SUGGESTIONS):
            if sug_cols[index % 3].button(
                suggestion, key=f"chat_sugg_{index}", width="stretch"
            ):
                pending = suggestion

        if st.button("🧹 مسح المحادثة | Clear chat", key="chat_clear", width="stretch"):
            state["chat_history"] = []
            st.rerun()

        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                meta = message.get("meta") or {}
                if message["role"] == "assistant" and meta.get("source") != "ai":
                    st.caption("ℹ️ إجابة محلية مبنية على أرقام بياناتك | Local data-grounded answer")

        question = st.chat_input(
            "اكتب سؤالك عن البيانات… | Ask about your data...", key="chat_input"
        )
        if question:
            pending = question

        if pending:
            history.append({"role": "user", "content": pending})
            with st.spinner("🔎 جاري تحليل السؤال... | Analyzing question..."):
                reply, meta = chat.answer(pending, ctx, api_key, model)
            history.append({"role": "assistant", "content": reply, "meta": meta})
            state["chat_history"] = history
            st.rerun()
