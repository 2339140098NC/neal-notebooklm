import os
from dotenv import load_dotenv
import fitz
from openai import OpenAI
from supabase import create_client

load_dotenv()

openai_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

doc = fitz.open("test.pdf")

#Get all the text from every page
full_text = ""
for page in doc:
    full_text += page.get_text()

#Chop into chunks of ~500 chars
chunk_size = 500
chunks = []
for i in range (0, len(full_text),chunk_size):
    chunks.append(full_text[i:i + chunk_size])

#Embed each chunk(turn each chunk into a vector)
response = openai_client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)

# Store in Supabase
for i, chunk in enumerate(chunks):
    supabase.table("documents").insert({
        "content": chunk,
        "embedding": response.data[i].embedding
    }).execute()
    
print(f"Stored {len(chunks)} chunks in Supabase!")

#print(f"Created {len(response.data)}")
#print(f"First vector preview: {response.data[0].embedding[:5]}")
#When you get the data back, each item had a property called .embedding that holds the actual vector(the list of number)
#So response.data[0] is the result for your first chunk, and .embedding reaches inside it to grab the vector specifically.