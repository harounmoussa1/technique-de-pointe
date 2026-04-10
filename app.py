import streamlit as st
from rag_chain import build_chain, ask
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="FallahTech Q&A — Comité d'investissement", layout="wide")
st.title("Assistant RAG — Dossier FallahTech Série A")
st.caption("Posez vos questions sur le dossier. Chaque réponse cite ses sources.")

if "chain" not in st.session_state:
    with st.spinner("Chargement de la base documentaire..."):
        st.session_state.chain = build_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Affichage historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources : " + " | ".join(msg["sources"]))

# Input
if question := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyse en cours..."):
            result = ask(st.session_state.chain, question, st.session_state.chat_history)

        st.markdown(result["answer"])
        if result["sources"]:
            st.caption("Sources : " + " | ".join(result["sources"]))

    # Mettre à jour l'historique LangChain
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=result["answer"]))

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
