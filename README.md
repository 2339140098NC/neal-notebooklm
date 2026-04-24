# Neal-Notebooklm

A PDF question-answering tool powered by Retrieval-Augmented Generation. Upload any PDF, ask questions, and get answers based on the document content.

![Demo](<I (1).gif>)

## How It Works

1. Upload a PDF → text is extracted and split into chunks
2. Each chunk is converted into a vector (embedding) using OpenAI
3. Chunks and vectors are stored in Supabase (pgvector)
4. When you ask a question, it finds the most relevant chunks by meaning
5. Those chunks + your question are sent to GPT, which answers based on the document

## Tech Stack

- **Python** — core language
- **FastAPI** — web server with API endpoints
- **OpenAI API** — embeddings (`text-embedding-3-small`) + chat completion (`gpt-4o-mini`)
- **Supabase (pgvector)** — vector storage and similarity search
- **PyMuPDF** — PDF text extraction
- **LangChain** — text splitting, vector store integration, and RAG chain (`RetrievalQA`)

## API Endpoints

- `POST /upload` — upload a PDF to process and store
- `POST /ask` — ask a question and get a RAG-powered answer

## Supabase Setup

Run the following in **Supabase → SQL Editor** before starting the server:

```sql
create extension if not exists vector;

create table documents (
  id uuid primary key default gen_random_uuid(),
  content text,
  metadata jsonb,
  embedding vector(1536)
);

create or replace function match_documents (
  query_embedding vector(1536),
  match_count int default null,
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

> If you previously created `match_documents` without the `filter` parameter, drop both overloads first:
> ```sql
> drop function if exists match_documents(vector, int);
> drop function if exists match_documents(vector, int, jsonb);
> ```

## How to Run

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn openai supabase pymupdf python-dotenv python-multipart langchain langchain-openai langchain-community langchain-text-splitters langchain-classic
   ```
3. Create a `.env` file (see `.env.example`)
4. Run the Supabase setup SQL above
5. Start the server: `uvicorn main:app --reload`
6. Visit `http://127.0.0.1:8000` to use the app

## Architecture Decisions

- **Why pgvector?** It lets us store vectors directly in PostgreSQL and do similarity search with a single query — no separate vector database needed.
- **Why 500-character chunks?** Small enough to be specific and relevant, large enough to carry meaningful context. A future improvement could split at sentence boundaries instead.
- **Why a grounded system prompt?** Without it, GPT blends its own knowledge with the PDF content. The prompt instructs it to answer only from the provided context, keeping responses grounded in the actual document.
- **Why k=10 retrieval?** Retrieving 10 chunks gives the model broader coverage of the document, reducing the chance of missing relevant content spread across multiple sections.

## Future Improvements

- Add a `source` column to track which PDF each chunk came from
- Smarter chunking (split at sentence boundaries instead of fixed length)
- Filter search results by specific document
