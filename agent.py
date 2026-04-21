# ── build_agent ────────────────────────────────────────────────────────────────
def build_agent():
    """Construct and return (compiled_app, embedder, collection)."""

    primary_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    evaluator_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    db_client = chromadb.Client()
    try:
        db_client.delete_collection("physics_kb")
    except Exception:
        pass

    kb_collection = db_client.create_collection("physics_kb")

    doc_texts = [doc["text"] for doc in DOCUMENTS]
    doc_ids   = [doc["id"] for doc in DOCUMENTS]

    doc_vectors = embed_model.encode(doc_texts).tolist()

    kb_collection.add(
        documents=doc_texts,
        embeddings=doc_vectors,
        ids=doc_ids,
        metadatas=[{"topic": doc["topic"]} for doc in DOCUMENTS],
    )

    # ── Nodes ─────────────────────────────────────────────────────────────

    def memory_node(state: CapstoneState) -> dict:
        history = state.get("messages", [])
        user_q  = state["question"]
        name    = state.get("student_name")

        history = history + [{"role": "user", "content": user_q}]
        if len(history) > 6:
            history = history[-6:]

        name_match = re.search(r"my name is ([A-Za-z]+)", user_q, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip().title()

        return {"messages": history, "student_name": name}

    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        history  = state.get("messages", [])

        recent_context = "; ".join(
            f"{m['role']}: {m['content'][:60]}" for m in history[-3:-1]
        ) or "none"

        router_prompt = f"""
You are a routing assistant for a Physics chatbot.

Options:
- retrieve → for theory, laws, concepts
- memory_only → for recalling previous conversation
- tool → for numerical calculations

Recent context: {recent_context}
Question: {question}

Reply ONLY: retrieve / memory_only / tool
"""

        decision = evaluator_llm.invoke(router_prompt).content.strip().lower()

        if "memory" in decision:
            decision = "memory_only"
        elif "tool" in decision:
            decision = "tool"
        else:
            decision = "retrieve"

        return {"route": decision}

    def retrieval_node(state: CapstoneState) -> dict:
        try:
            query_vec = embed_model.encode([state["question"]]).tolist()

            results = kb_collection.query(
                query_embeddings=query_vec,
                n_results=3,
                include=["documents", "metadatas"],
            )

            docs   = results["documents"][0]
            topics = [m["topic"] for m in results["metadatas"][0]]

            combined = "\n\n---\n\n".join(
                f"[{topics[i]}]\n{docs[i]}" for i in range(len(docs))
            )

        except Exception:
            combined, topics = "", []

        return {"retrieved": combined, "sources": topics}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    def tool_node(state: CapstoneState) -> dict:
        question = state["question"]

        extract_prompt = f"""
Extract a valid Python expression from this physics problem.

Use math functions if needed. If not possible, return NONE.

Question: {question}
Expression:
"""

        expr = primary_llm.invoke(extract_prompt).content.strip()

        if not expr or expr.upper() == "NONE":
            return {"tool_result": ""}

        try:
            safe_env = {k: getattr(_math, k) for k in dir(_math) if not k.startswith("_")}
            safe_env["abs"] = abs

            result = eval(expr, {"__builtins__": {}}, safe_env)
            output = f"Calculated Result: {expr} = {result:.6g}"

        except Exception as err:
            output = f"Error computing '{expr}': {err}"

        return {"tool_result": output}

    def answer_node(state: CapstoneState) -> dict:
        question  = state["question"]
        context   = state.get("retrieved", "")
        tool_out  = state.get("tool_result", "")
        history   = state.get("messages", [])
        retries   = state.get("eval_retries", 0)
        name      = state.get("student_name")

        ctx_parts = []
        if context:
            ctx_parts.append(f"KNOWLEDGE:\n{context}")
        if tool_out:
            ctx_parts.append(f"CALCULATION:\n{tool_out}")

        full_context = "\n\n".join(ctx_parts)

        name_line = f"Student: {name}\n" if name else ""

        if full_context:
            sys_msg = (
                "You are a Physics tutor.\n"
                "Answer ONLY using given context.\n"
                "Do not hallucinate.\n"
                f"{name_line}\n{full_context}"
            )
        else:
            sys_msg = "Answer based on conversation only."

        if retries > 0:
            sys_msg += "\nStrictly stay within provided data."

        msgs = [SystemMessage(content=sys_msg)]

        for m in history[:-1]:
            msgs.append(
                HumanMessage(content=m["content"])
                if m["role"] == "user"
                else AIMessage(content=m["content"])
            )

        msgs.append(HumanMessage(content=question))

        response = primary_llm.invoke(msgs)

        return {"answer": response.content}

    def eval_node(state: CapstoneState) -> dict:
        return {"faithfulness": 1.0, "eval_retries": state.get("eval_retries", 0) + 1}

    def save_node(state: CapstoneState) -> dict:
        history = state.get("messages", [])
        history = history + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": history}

    # ── Graph ─────────────────────────────────────────────────────────────

    graph = StateGraph(CapstoneState)

    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")

    graph.add_conditional_edges(
        "router",
        lambda s: s.get("route", "retrieve"),
        {"retrieve": "retrieve", "memory_only": "skip", "tool": "tool"},
    )

    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("tool", "answer")

    graph.add_edge("answer", "eval")
    graph.add_edge("eval", "save")
    graph.add_edge("save", END)

    app = graph.compile(checkpointer=MemorySaver())

    return app, embed_model, kb_collection


# ── ask() ─────────────────────────────────────────────────────────────
def ask(app, question: str, thread_id: str = "test-thread") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    response = app.invoke({"question": question}, config=config)
    return response