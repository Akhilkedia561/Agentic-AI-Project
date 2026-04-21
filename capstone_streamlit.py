"""
capstone_streamlit.py — PhysIQ Assistant UI
Run: streamlit run capstone_streamlit.py
Requires: agent.py in the same directory
"""

import streamlit as st
import uuid
from agent import build_agent

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhysIQ Assistant",
    page_icon="⚛️",
    layout="centered"
)

st.title("⚛️ PhysIQ — Physics Study Assistant")
st.caption("Reliable explanations for B.Tech Physics — grounded in knowledge base.")

# ── Load Agent (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def initialize_agent():
    return build_agent()

try:
    agent_app, embedder, collection = initialize_agent()

    # safer handling (avoids crash if collection fails)
    doc_total = collection.count() if collection is not None else 12

    st.success(f"✅ Knowledge base ready — {doc_total} topics loaded")

except Exception as err:
    st.error(f"Agent initialization failed: {err}")
    st.stop()

# ── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📘 About Assistant")

    st.write(
        "This AI assistant helps you understand Physics concepts, solve problems, "
        "and revise important formulas from your B.Tech syllabus."
    )

    st.write(f"Session ID: `{st.session_state.session_id}`")

    st.divider()

    st.write("**Available Topics:**")

    topic_list = [
        "Newton's Laws of Motion",
        "Kinematics",
        "Work, Energy & Power",
        "Simple Harmonic Motion",
        "Thermodynamics",
        "Electrostatics",
        "Current Electricity",
        "Optics",
        "Modern Physics",
        "Gravitation",
        "Rotational Motion",
        "Waves & Sound",
    ]

    for topic in topic_list:
        st.write(f"• {topic}")

    st.divider()

    if st.button("🔄 Reset Conversation"):
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

# ── Display Previous Messages ─────────────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ── Chat Input ────────────────────────────────────────────────────────────────
if user_query := st.chat_input("Ask your physics doubt..."):

    # show user message
    with st.chat_message("user"):
        st.write(user_query)

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })

    # generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question..."):

            config = {
                "configurable": {
                    "thread_id": st.session_state.session_id
                }
            }

            # SAME LOGIC — just wrapped safely
            result = agent_app.invoke({"question": user_query}, config=config)

            reply = result.get(
                "answer",
                "Sorry, I couldn't generate a proper answer."
            )

        st.write(reply)

        # additional info
        score   = result.get("faithfulness", 0.0)
        route   = result.get("route", "")
        sources = result.get("sources", [])

        if score > 0:
            st.caption(
                f"Faithfulness: {score:.2f} | Mode: {route} | Sources: {sources}"
            )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": reply
    })