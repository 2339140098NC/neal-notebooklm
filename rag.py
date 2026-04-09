import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

#1. Ask a question
question = input("Ask a question about the PDF: ")

#2. Turn question into a vector
q_response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=question
)

q_vector = q_response.data[0].embedding

#2. Find most similar chunks in Supabase
results = supabase.rpc("match_documents",{
    "query_embedding": q_vector,
    "match_count":3
}).execute()

chunks = [r["content"] for r in results.data]
context = "\n\n".join(chunks)

#Send chunks + question to AI
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
        {"role": "system", "content":f"Answer based on this context: \n\n{context}"},
        {"role":"user", "content": question}
    ]
)

print("\n" + response.choices[0].message.content)