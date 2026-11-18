# Healthcare Policy Assistant

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload a healthcare policy PDF (e.g. a CMS Medicare policy document) and then:

- Ask questions about the document (Q&A)
- Summarize the entire policy
- Extract all rules and requirements

Built with **Streamlit**, **LangChain**, **FAISS**, **HuggingFace embeddings**, and **Mistral AI** as the LLM.

---

## Features

| Tab | Description |
|---|---|
| **Q&A** | Ask any question about the uploaded policy document and get an answer grounded strictly in its content. |
| **Summarize** | Generates a concise summary of the entire policy document. |
| **Extract Rules** | Lists all rules and requirements found in the policy document. |

The assistant only answers using the content of the uploaded PDF. If the answer isn't found in the document, it will let you know instead of hallucinating a response.

---

## Project Structure

```
├── healthcare_policy_assistant_backend.py   # Document loading, chunking, embeddings, vector store, QA chain
├── healthcare_policy_assistant_frontend.py  # Streamlit UI
├── requirements.txt                         # Python dependencies
├── .env                                      # Your local secrets (NOT committed to GitHub)
```

---

## Prerequisites

- Python 3.9+
- A [Mistral AI](https://console.mistral.ai/) API key

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vidhya-95/healthcare_policy_assistant.git
cd healthcare_policy_assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root with your Mistral API key:

```
MISTRAL_API_KEY=your_mistral_api_key_here
```



### 4. Run the app

```bash
streamlit run healthcare_policy_assistant_frontend.py
```

The app will open in your browser. Upload a policy PDF and start asking questions.

---

## How It Works

1. **Load & Chunk** – The uploaded PDF is parsed with `PyPDFLoader` and split into overlapping chunks using `RecursiveCharacterTextSplitter` for better context retention.
2. **Embed & Store** – Chunks are embedded using the `sentence-transformers/all-MiniLM-L6-v2` model and stored in a local **FAISS** vector store.
3. **Retrieve & Answer** – On each query, the top matching chunks are retrieved and passed, along with a system prompt, to **Mistral's `mistral-small-latest`** model via `ChatMistralAI` to generate a grounded answer.

---

## Tech Stack

- [Streamlit](https://streamlit.io/) – Web UI
- [LangChain](https://www.langchain.com/) – Orchestration (document loading, chunking, retrieval chains)
- [FAISS](https://github.com/facebookresearch/faiss) – Vector similarity search
- [Sentence Transformers](https://www.sbert.net/) – Embeddings
- [Mistral AI](https://mistral.ai/) – LLM for answer generation

---

## Notes

- The app currently processes one PDF at a time; re-uploading a new PDF rebuilds the vector store from scratch.
- Answers are strictly grounded in the uploaded document — the assistant is instructed not to answer from general knowledge.

---

