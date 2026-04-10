import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = "./chroma_db"

SYSTEM_PROMPT = """Tu es un assistant d'analyse financière pour un comité d'investissement.
Tu analyses le dossier de la startup FallahTech SARL (AgriTech tunisienne, Série A).

Règles strictes :
1. Tu réponds UNIQUEMENT en te basant sur les documents fournis ci-dessous.
2. Chaque affirmation doit citer sa source entre crochets, ex: [source: bilan_2025.pdf]
3. Si l'information est absente des documents, dis : "Cette information n'est pas disponible dans le dossier fourni."
4. Ton ton est professionnel, adapté à un comité d'investissement.

Documents pertinents :
{context}
"""

_retriever = None

def build_chain():
    global _retriever

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    _retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[source: {doc.metadata.get('source', 'inconnu')}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(_retriever.invoke(x["input"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask(chain, question: str, chat_history: list) -> dict:
    answer = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    docs = _retriever.invoke(question)
    sources = list({doc.metadata.get("source", "inconnu") for doc in docs})

    return {
        "answer": answer,
        "sources": sources
    }
