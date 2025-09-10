import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # To allow browser requests

#from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------------
# Load Environment and Initialize Services
# ------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# Initialize LLM
llm = ChatOpenAI(api_key=api_key, model_name="gpt-4o-mini")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Load Vector Store
DB_FAISS_PATH = "vectorstore/db_faiss"
if not os.path.exists(DB_FAISS_PATH):
    print("Vector store not found. Please run your original script once to create it.")
    exit() # Exit if the vector store doesn't exist

print("Loading existing vector store...")
vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
print("Vector store loaded.")

# Prompt Template
prompt_template_string = """
You are a professional assistant for SaaviGenAI.
Answer the user's question based ONLY on the provided context.
If the context does not contain the answer, politely state that you do not have that information.
Be concise and clear.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(
    input_variables= ["context", "question"],
    template= prompt_template_string
)

# Build RAG Chain
rag_chain = prompt | llm | StrOutputParser()

# ------------------------
# FastAPI Application
# ------------------------
app = FastAPI()

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the request model
class ChatRequest(BaseModel):
    question: str
    # We won't manage history on the backend for this simple static page example
    # A more complex app would use sessions or a database for history

@app.post("/chat")
def chat_with_bot(request: ChatRequest):
    """
    Receives a question, gets context from FAISS, and returns an answer from the RAG chain.
    """
    try:
        # Retrieve relevant docs
        docs = vector_store.similarity_search(request.question, k=2)
        context_text = "\n".join([doc.page_content for doc in docs])

        # Prepare inputs
        inputs = {
            "question": request.question,
            "context": context_text
        }

        # Invoke the RAG chain
        result = rag_chain.invoke(inputs)
        return {"answer": result}

    except Exception as e:
        print("Error during chat processing:", e)
        return {"answer": "Sorry, an error occurred while processing your request."}

# To run this, use the command in your terminal:
# uvicorn backend:app --reload