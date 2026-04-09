# Neal-Notebooklm

A PDF question-answering tool powered by Retrieval-Augmented Generation. Upload any PDF, ask questions, and get answers based on the document content.

## How It Works

1. Upload a PDF → text is extracted and split into chunks
2. Each chunk is converted into a vector (embedding) using OpenAI
3. Chunks and vectors are stored in Supabase (pgvector)
4. When you ask a question, it finds the most relevant chunks by meaning
5. Those chunks + your question are sent to GPT, which answers based on the document

## Tech Stack

- **Python** — core language
- **FastAPI** — web server with API endpoints
- **OpenAI API** — embeddings + chat completion
- **Supabase (pgvector)** — vector storage and similarity search
- **PyMuPDF** — PDF text extraction

## API Endpoints

- `POST /upload` — upload a PDF to process and store
- `POST /ask` — ask a question and get a RAG-powered answer

## How to Run

1. Clone the repo
2. Install dependencies: `pip install fastapi uvicorn openai supabase pymupdf python-dotenv python-multipart`
3. Create a `.env` file (see `.env.example`)
4. Start the server: `python -m uvicorn main:app --reload`
5. Visit `http://127.0.0.1:8000/docs` to test

## Architecture Decisions

- **Why pgvector?** It lets us store vectors directly in PostgreSQL and do similarity search with a single query — no separate vector database needed.
- **Why 500-character chunks?** Small enough to be specific and relevant, large enough to carry meaningful context. A future improvement could split at sentence boundaries instead.
- **Why a stricter system prompt?** Without it, GPT blends its own knowledge with the PDF content. Adding "answer ONLY based on this context" keeps responses grounded in the actual document.

## Future Improvements

- Add a `source` column to track which PDF each chunk came from
- Smarter chunking (split at sentence boundaries instead of fixed length)
- Filter search results by specific document
- Add a frontend UI
