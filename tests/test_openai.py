from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openai_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [{"role": "user", "content": "What is the capital of France?"}]
)

print(response.choices[0].message.content)