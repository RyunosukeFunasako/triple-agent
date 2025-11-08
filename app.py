import streamlit as st

# description_page = st.Page("description.py", title="実験の説明", icon="✅")
# qids_page = st.Page("before-session/qids.py", title="抑うつ症状の検査", icon="😔")
# attention_page = st.Page("attention.py", title="対話セッションを始める前に", icon="👀")
dialogue_page = st.Page("dialogue-session/dialogue.py", title="対話セッション", icon="👩‍⚕️")
# cc_immediate_page = st.Page("after-session/cc_immediate.py", title="認知変化の評価", icon="🧠")
# rapport_page = st.Page("after-session/rapport.py", title="カウンセラーの評価", icon="📄")
# quality_page = st.Page("after-session/quality.py", title="システムの評価", icon="💬")
# thanks_page = st.Page("thanks.py", title="実験終了", icon="👏")


if "current_page" not in st.session_state:
    # st.session_state.current_page = "description"
    st.session_state.current_page = "dialogue"

# # QIDS記録
# if "qids_answers" not in st.session_state:
#     st.session_state.qids_answers = {}

# 対話履歴記録
if "dialogue_history" not in st.session_state:
    st.session_state.dialogue_history = []

# 逸脱履歴記録
if "deviation_history" not in st.session_state:
    st.session_state.deviation_history = []


# # CC-immediate記録
# if "cc_immediate_answers" not in st.session_state:
#     st.session_state.cc_immediate_answers = {}

# # rapport記録
# if "rapport_answers" not in st.session_state:
#     st.session_state.rapport_answers = {}

# # quality記録
# if "quality_answers" not in st.session_state:
#     st.session_state.quality_answers = {}

# if st.session_state.current_page == "description":
#     pg = st.navigation([description_page])
# elif st.session_state.current_page == "qids":
#     pg = st.navigation([qids_page])
# elif st.session_state.current_page == "attention":
#     pg = st.navigation([attention_page])
if st.session_state.current_page == "dialogue":
    pg = st.navigation([dialogue_page])
# elif st.session_state.current_page == "cc_immediate":
#     pg = st.navigation([cc_immediate_page])
# elif st.session_state.current_page == "rapport":
#     pg = st.navigation([rapport_page])
# elif st.session_state.current_page == "quality":
#     pg = st.navigation([quality_page])
# elif st.session_state.current_page == "thanks":
#     pg = st.navigation([thanks_page])

pg.run()