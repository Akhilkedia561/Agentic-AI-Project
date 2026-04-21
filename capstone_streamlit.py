"""
capstone_streamlit.py — Physics Study Buddy Agent
Run: streamlit run capstone_streamlit.py
Requires: agent.py in the same directory
"""
import streamlit as st
import uuid
from agent import build_agent

st.set_page_config(page_title="Physics Study Buddy", page_icon="⚛️", layout="centered")
st.title("⚛️ Physics Study Buddy")
st.caption("Explains B.Tech Physics concepts faithfully — no hallucinated formulas.")


@st.cache_resource
def load_agent():
    return build_agent()


try:
    agent_app, embedder, collection = load_agent()
    st.success(f"✅ Knowledge base loaded — {collection.count()} documents")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

with st.sidebar:
    st.header("About")
    st.write(
        "Ask about Physics concepts, laws, formulas, and numerical problems "
        "from your B.Tech curriculum. The agent retrieves answers from its "
        "curated knowledge base and uses a built-in calculator for numerical queries."
    )
    st.write(f"Session ID: `{st.session_state.thread_id}`")
    st.divider()
    st.write("**Topics covered:**")
    topics = [
        "Newton\'s Laws of Motion", "Kinematics — Equations of Motion",
        "Work, Energy, and Power", "Simple Harmonic Motion",
        "Thermodynamics", "Electrostatics",
        "Current Electricity", "Optics — Reflection & Refraction",
        "Modern Physics", "Gravitation",
        "Rotational Motion", "Waves and Sound",
    ]
    for t in topics:
        st.write(f"• {t}")
    st.divider()
    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a physics question..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, I could not generate an answer.")
        st.write(answer)
        faith   = result.get("faithfulness", 0.0)
        route   = result.get("route", "")
        sources = result.get("sources", [])
        if faith > 0:
            st.caption(f"Faithfulness: {faith:.2f} | Route: {route} | Sources: {sources}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
