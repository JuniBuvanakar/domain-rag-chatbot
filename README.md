# Domain-Specific RAG Chatbot

An AI-powered chatbot that answers questions from educational PDF documents using Retrieval-Augmented Generation (RAG).

## Features
- Upload educational PDF documents.
- Ask questions in natural language.
- Retrieve relevant information using FAISS vector search.
- Generate answers using the Qwen3:4b model through Ollama.
- Display source PDF filenames and page numbers.
- Handle questions whose answers are not found in the documents.
- Clear chat history.

## Technologies Used
- Python
- Streamlit
- Ollama
- Qwen3:4b
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- PyPDF
- LangChain Text Splitters

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/JuniBuvanakar/domain-rag-chatbot.git
cd domain-rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the model
```bash
ollama pull qwen3:4b
```

### 4. Run the application
```bash
streamlit run app.py
```

Make sure Ollama is running before starting the application.

## Project Objective

To help students retrieve information from educational documents quickly by combining semantic search with a locally running large language model.

## Author
**V.A. JUNI BUVANAKAR**

GitHub: https://github.com/JuniBuvanakar
