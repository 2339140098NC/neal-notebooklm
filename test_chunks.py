import fitz

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

print(f"Total chunks: {len(chunks)}")
print(f"\n--- First chunk ---\n{chunks[0]}")
print(f"\n--- Second chunk ---\n{chunks[1]}")