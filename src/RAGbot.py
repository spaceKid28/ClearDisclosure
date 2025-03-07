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
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer  # Import GPT-2 model and tokenizer from HuggingFace (I didn't want to pay for tokens so decided to use GPT-2)

class RAGbot:
    # class variables that are accessible by all methods
    documents = [] # class variable which will contain strings from pdfs as key value {text : pdf text, source : filename of source (ie 2024 Walmart 10k)}
    chunks = [] # class variable which will contain all the chunks
    # Load embedding model from HuggingFace: https://huggingface.co/sentence-transformers/
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # I used the smallest model so that I could run it locally on my machine.    
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Initialize the GPT-2 tokenizer
    llm = GPT2LMHeadModel.from_pretrained("gpt2")  # Initialize the GPT-2 model

    def __init__(self, pdf_directory="./data/"):
        # Directory containing your PDFs, by default it is labeled data and in the same working directory as /src/
        self.pdf_directory = pdf_directory
        
    def clean(self):
        self.write_pdfs_to_strings() # this method writes to documents class variable
        self.chunking()
        self.create_embeddings()

    def write_pdfs_to_strings(self):
        # Process all PDFs
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_directory, filename)
                text = self.extract_text_from_pdf(pdf_path)
                self.documents.append({"text": text, "source": filename})
        

    def extract_text_from_pdf(self, pdf_path: str) -> str: # helper method for write_pdfs_to_strings, returns text from a pdf file as a string
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages: # loop through each page of pdf
            text += page.extract_text() + "\n"
        return text
    
    def chunking(self): # to improve the performance of our RAG agent, we are going to split up the text into chunks of 1000 characters
        # CHUNKING
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        for doc in self.documents: # loop through the documents (which have already been converted from PDFs into strings)
            texts = text_splitter.split_text(doc["text"])
            for text in texts: # loop through chunks, and keep track of souce text (so when RAG agent uses a chunk, we know from which PDF it pulled from)
                self.chunks.append({
                    "text": text,
                    "source": doc["source"]
                })

    def create_embeddings(self):
        # convert the text chunk into vector embeddings
        # this is done via sentence_transformers api

        embeddings = [] # I think this can be deleted
        texts = [chunk["text"] for chunk in self.chunks] # just creating a list of strings, recall chunks = [{text: "pdf string", source: pdf_filename}]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True) # Generate embeddings (This converts strings into vectors)
        embeddings_np = np.array(embeddings).astype('float32') # Convert embeddings to float32 numpy array
        dimension = embeddings_np.shape[1] # Get embedding dimension

        # Create FAISS index, more information can be found here https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexFlatL2.html
        index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        index.add(embeddings_np) # FAISS allows us to calculate similarity between vectors (I've chosen to use Euclidian Distance, but we could tinker and see which gives us the best results (Manhattan, Cosine, ect...))
        
        faiss.write_index(index, "./output/pdf_embeddings.index") # we save the embeddings, which are numeric vectors

        # Save chunks data separately (to maintain text-embedding, remember that later on we will want to know which Documents our RAG agent used to generate its responses)
        with open("./output/pdf_chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def retrieve_documents(self, query : str, top_k : int = 3) -> list: # helper function, for answer question. Returns a list of the documents to use in response, based on prompt
        index = faiss.read_index("./output/pdf_embeddings.index") # read in the embeddings (vectors)
        with open("./output/pdf_chunks.pkl", "rb") as f: # read in the actual text of the pdfs
            chunks = pickle.load(f)
        
        query_embedding = self.embedding_model.encode([query]) # This converts our user-defined query into a vector so we can perform similarity search and pull the correct documents
        
        distances, indices = index.search(query_embedding, top_k) # find the top_k chunks of text that are most similar to our query
        
        # indicies give us the indicies of the top_k chunks that are most similar to the prompt. If we had 100 chunks, and top_k = 3, it could look like [[89, 34, 2]]
        # We will loop through indicies, and pull the relevant chunks of TEXT, this is why we pickled the text and each chunk's source PDF)
        # distance shows us how similar each chunk is to the user-defined query
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "distance": distances[0][i]
            })
        
        return results

    def answer_question(self, query):
        # Retrieve relevant documents
        docs = self.retrieve_documents(query)
        
        # Use the text that we pulled (based on similarity to prompt) as context for the LLM
        context = f"Results for query: '{query}'\n\n"
        for i, doc in enumerate(docs, 1): # Change 0 index to start at 1 index
            context += f"Result {i} (from {doc['source']}):\n{doc['text']}\n\n"
        
        print(f"CONTEXT: {context}\n\n")
        # Use the LLM to generate a response based on the context
        # IMPORTANT: Because I am using chat-gpt2, I must limit my input to 1000 tokens... would improve if I could provide more context
        # but I want it to run locally, so that is the compromise I made

        # Set pad_token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
            # Ensure we're not exceeding model's position embedding limit
        inputs = self.tokenizer.encode(context, return_tensors="pt", truncation=True, max_length=900)
        
        # Generate response
        outputs = self.llm.generate(
            inputs,
            max_new_tokens=150,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract only the new tokens (not the input)
        input_length = inputs.shape[1]
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        return response

 