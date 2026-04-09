import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

response = openai_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY")).embeddings.create(
    model = "text-embedding-3-small",
    input = "How's life?"
)

vector = response.data[0].embedding
print(f"Number of dimensions: {len(vector)}")
print(f"First 5 numbers: {vector[:5]}")
