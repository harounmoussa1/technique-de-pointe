# Assistant RAG — Dossier FallahTech Série A

Chatbot de question-réponse sur un data room d'investissement. Pose une question en langage naturel, l'assistant répond en citant ses sources.

---

## Stack technique

| Composant | Technologie |
|---|---|
| LLM | Meta LLaMA 3.3 70B via API Groq |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace, local) |
| Base vectorielle | ChromaDB |
| Framework | LangChain |
| Interface | Streamlit |

---

## Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/harounmoussa1/technique-de-pointe.git
cd technique-de-pointe
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Configurer la clé API
Crée un fichier `.env` à la racine du projet :
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```
> Clé gratuite sur [console.groq.com](https://console.groq.com)

### 4. Ajouter les documents
Place tes PDFs et fichiers Excel dans le dossier `documents/` (sous-dossiers acceptés).

### 5. Indexer les documents (une seule fois)
```bash
python ingest.py
```

### 6. Lancer le chatbot
```bash
streamlit run app.py
```
Ouvre **http://localhost:8501** dans le navigateur.

---

## Structure du projet

```
├── documents/          ← PDFs et Excel à indexer
├── chroma_db/          ← Base vectorielle (générée par ingest.py)
├── ingest.py           ← Indexation des documents
├── rag_chain.py        ← Chaîne RAG (retrieval + LLM)
├── app.py              ← Interface Streamlit
├── requirements.txt
└── .env                ← GROQ_API_KEY (non versionné)
```

---

## Utilisation

1. Lance `streamlit run app.py`
2. Pose ta question dans le champ de saisie
3. La réponse cite automatiquement les documents sources
