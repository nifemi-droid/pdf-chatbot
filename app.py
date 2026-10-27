import os
from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import io

app = FastAPI()

# Initialize our tools
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()

# We will store our collection here
collection = chroma_client.get_or_create_collection(name="pdf_documents")

client = Groq(api_key="YOUR_GROQ_API_KEY") # <--- PASTE KEY HERE

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. Read the PDF
    contents = await file.read()
    pdf = PdfReader(io.BytesIO(contents))
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
        
    # 2. Chunk the text (simple split by paragraph)
    chunks = [line.strip() for line in text.split('\n') if len(line.strip()) > 50]
    
    if not chunks:
        return {"error": "Could not extract text from PDF. Is it a scanned image?"}
    
    # 3. Embed the chunks
    embeddings = embedding_model.encode(chunks).tolist()
    
    # 4. Store in ChromaDB (we clear it first so it acts like a new session)
    chroma_client.delete_collection(name="pdf_documents")
    collection = chroma_client.create_collection(name="pdf_documents")
    
    # Generate IDs for the chunks
    ids = [str(i) for i in range(len(chunks))]
    
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids
    )
    
    return {"message": f"Successfully ingested {len(chunks)} chunks from {file.filename}"}

@app.get("/chat")
async def chat(query: str):
    # 1. Embed the user's question
    query_embedding = embedding_model.encode([query]).tolist()
    
    # 2. Search ChromaDB for the 2 most relevant chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    
    # 3. Extract the text from the results
    context = "\n".join(results['documents'][0])
    
    # 4. Ask Groq
    prompt = f"""
    You are a helpful assistant. Answer the user's question using ONLY the context below.
    If the answer is not in the context, say "I don't know."
    
    Context: {context}
    
    Question: {query}
    """
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b", # Using the cheap model we proved works!
    )
    
    return {"answer": response.choices[0].message.content}