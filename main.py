import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_pdf(file: UploadFile):
    #Read PDF
    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    #Chunk
    chunk_size = 500
    chunks = []
    for i in range(0, len(full_text), chunk_size):
        chunks.append(full_text[i:i + chunk_size])

    #Embed
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )

    #Store
    for i,chunk in enumerate(chunks):
        supabase.table("documents").insert({
            "content": chunk,
            "embedding": response.data[i].embedding
        }).execute()

    return {"message": f"Stored {len(chunks)} chunks"}

@app.post("/ask")
async def ask_question(question: str):
    #Embed question
    q_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    q_vector = q_response.data[0].embedding

    #Search
    results = supabase.rpc("match_documents",{
        "query_embedding": q_vector,
        "match_count":5
    }).execute()

    chunks = [r["content"] for r in results.data]
    context= "\n\n".join(chunks)

    #Send chunks + question to AI
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "system", "content": f"Answer ONLY based on this context. If the answer is not in the context, say 'I don't have that information.'\n\n{context}"},
            {"role":"user", "content": question}
        ]
    )

    return {"answer": response.choices[0].message.content}
