import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import io

# Load environment variables (this reads the GROQ_API_KEY from the .env file)
load_dotenv()

app = FastAPI()

# Initialize our tools
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. Read the PDF
    contents = await file.read()
    pdf = PdfReader(io.BytesIO(contents))
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
        
    # 2. Chunk the text (split by paragraph, ignore tiny chunks)
    chunks = [line.strip() for line in text.split('\n') if len(line.strip()) > 50]
    
    if not chunks:
        return {"error": "Could not extract text from PDF. Is it a scanned image?"}
    
    # 3. Embed the chunks
    embeddings = embedding_model.encode(chunks).tolist()
    
    # 4. Store in ChromaDB (clear old data first)
    try:
        chroma_client.delete_collection(name="pdf_documents")
    except Exception:
        pass # If it doesn't exist, do nothing
        
    global collection
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
    try:
        # 1. Get the active collection (fixes the previous bug)
        current_collection = chroma_client.get_collection(name="pdf_documents")
        
        # 2. Embed the user's question
        query_embedding = embedding_model.encode([query]).tolist()
        
        # 3. Search ChromaDB for the 2 most relevant chunks
        results = current_collection.query(
            query_embeddings=query_embedding,
            n_results=2
        )
        
        # 4. Extract the text from the results
        context = "\n".join(results['documents'][0])
        
        # 5. Ask Groq
        prompt = f"""
        You are a helpful assistant. Answer the user's question using ONLY the context below.
        If the answer is not in the context, say "I don't know."
        
        Context: {context}
        
        Question: {query}
        """
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
        )
        
        return {"answer": response.choices[0].message.content}
        
    except Exception as e:
        # This will help us see the exact error if it fails again
        print(f"BACKEND ERROR: {str(e)}")
        return {"answer": f"An error occurred: {str(e)}"}