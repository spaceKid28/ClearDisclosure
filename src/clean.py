import os
# this is for converting the PDFs into strings (technically a list of dictionaries)
from PyPDF2 import PdfReader
# this if for chunking... langchain provides this api, which should improve the retrevial component of my RAG implementation
from langchain.text_splitter import RecursiveCharacterTextSplitter
# after chunking, we must convert the text chunks into vector embeddings... the vector embeddings are necessary
# because they are how the RAG model performs similarlity comparison with the prompt and the text in the database
from sentence_transformers import SentenceTransformer
# also we have to store the embeddings somewhere, so I put them in FAISS, which a open source vector database published by Facebook
import faiss
import numpy as np
# pickle is python package to write data 
import pickle


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Directory containing your PDFs
pdf_directory = "../data/"
documents = []

# Process all PDFs
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        text = extract_text_from_pdf(pdf_path)
        documents.append({"text": text, "source": filename})

for hashmap in documents:
    print(f"First 10 characters of cleaned Text: {hashmap['text'][:10]}, Document name: {hashmap['source']}")


# CHUNKING
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = []
for doc in documents:
    texts = text_splitter.split_text(doc["text"])
    for text in texts:
        chunks.append({
            "text": text,
            "source": doc["source"]
        })
for chunk in chunks:
    print(f"First 10 characters of chunks: {chunk['text'][:10]}, Document name: {chunk['source']}")

# convert the text chunk into vector embeddings
# this is done via sentence_transformers api, I used the smallest model 
# so that I could run it locally on my machine. 

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model

# Generate embeddings
embeddings = []
texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

# Convert embeddings to float32 numpy array
embeddings_np = np.array(embeddings).astype('float32')

# Get embedding dimension
dimension = embeddings_np.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
index.add(embeddings_np)

# Save index to disk
faiss.write_index(index, "pdf_embeddings.index")

# TO DO I DON'T UNDERSTAND WHY WE NEED THE PICKLE FILES

# Save chunks data separately (to maintain text-embedding relationship)
with open("pdf_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

def retrieve_documents(query, top_k=5):
    # Load index and chunks
    index = faiss.read_index("pdf_embeddings.index")
    with open("pdf_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    
    # Encode query
    
    query_embedding = model.encode([query])
    
    # Search index
    distances, indices = index.search(query_embedding, top_k)
    
    # Return relevant chunks
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "distance": distances[0][i]
        })
    
    return results

query = "What are some strategic risks that are highlighted in Walmart's 2024 10-k?"

from langchain.llms import HuggingFaceHub  # Or use any other LLM

def answer_question(query):
    # Retrieve relevant documents
    docs = retrieve_documents(query)
    
    # Create context from retrieved documents
    context = "\n\n".join([doc["text"] for doc in docs])
    
    # Create prompt with context
    prompt = f"""Answer the question based on the following context:

            Context:
            {context}

            Question: {query}

            Answer:"""
                
                # Get response from LLM
                # (Replace with your preferred LLM interface)
                llm = HuggingFaceHub(repo_id="google/flan-t5-large")
                response = llm(prompt)
                
                return {
                    "answer": response,
                    "sources": [doc["source"] for doc in docs]
                }