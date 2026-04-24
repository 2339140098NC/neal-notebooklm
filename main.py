import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50 #repeat 50 chars between chunks so nothing gets cut off
    )
    chunks = splitter.create_documents([full_text])

    #Embed + Store
    #Embed + Store in batches
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    batch_size = 50

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        SupabaseVectorStore.from_documents(
            batch,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )
        print(f"Stored batch {i//batch_size + 1}")

    return {"message": f"Stored {len(chunks)} chunks"}

@app.post("/ask")
async def ask_question(question:str):
    vectorstore = SupabaseVectorStore(
        client=supabase,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        table_name = "documents",
        query_name="match_documents"
    )

        # DEBUG: see what chunks are being retrieved
    docs = vectorstore.similarity_search(question, k=10)
    print(f"Found {len(docs)} chunks")
    for doc in docs:
        print(doc.page_content[:100])

    prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following context to answer the question. 
        If the answer is not in the context, say 'I don't have that information.'

        Context:
        {context}

        Question: {question}
        Answer:"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain.invoke({"query":question})
    return {"answer": result["result"]}
