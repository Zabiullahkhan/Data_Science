import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from ollama import chat
from ollama import ChatResponse

# Initialize the Sentence-Transformers embeddings
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define the function to load, split, and vectorize the PDF content
def load_split_vec(input_document):
    loader = PyPDFLoader(input_document, extract_images=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    # Store the split documents in the FAISS vector store
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("FAISS_Vectorstore.db")
    print("Documents processed and saved in vector database.")

# Define the function to query the vector store and get a response
def llm_run(question):
    new_db = FAISS.load_local("FAISS_Vectorstore.db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    
    # Constructing the prompt chain for the question-answering
    client = chat(model="smollm:1.7b", messages=[{"role": "user", "content": question}])
    relevant_documents = retriever.invoke(question)
    
    # Format relevant documents text for the LLM
    doc_texts = "\n".join([doc.page_content for doc in relevant_documents])
    
    # Get the response from the model
    response = client["message"]["content"]
    return response

# Function to handle user input and query the RAG system
def main():
    print("Welcome to the RAG-based Chat with PDF system!")
    
    # Ask the user to input the path to the PDF document
    input_pdf = input("Please enter the path to your PDF document: ")
    
    if not os.path.exists(input_pdf):
        print("The file does not exist. Please check the file path.")
        return
    
    # Process the PDF file
    load_split_vec(input_pdf)
    
    # Ask the user for a question
    while True:
        user_question = input("\nPlease ask a question (or type 'exit' to quit): ")
        
        if user_question.lower() == 'exit':
            print("Exiting the system.")
            break
        
        # Query the model and get the response
        response = llm_run(user_question)
        
        # Print the response
        print(f"Response: {response}")

if __name__ == "__main__":
    main()



################################################################### WITH STREAMLIT ##################################################################
# utils/document_processor.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import tempfile

class DocumentProcessor:
    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    
    def process_pdf(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            documents = PyPDFLoader(tmp_file.name, extract_images=True).load()
        os.unlink(tmp_file.name)
        
        split_docs = self.text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        vectorstore.save_local("FAISS_Vectorstore.db")
        return True

# utils/chat_manager.py
from ollama import chat
from langchain.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

class ChatManager:
    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
    def get_response(self, question):
        vectorstore = FAISS.load_local("FAISS_Vectorstore.db", self.embeddings, allow_dangerous_deserialization=True)
        relevant_docs = vectorstore.as_retriever().invoke(question)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""Context: {context}\n\nQuestion: {question}"""
        response = chat(model="smollm:1.7b", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"], context

# app.py
import streamlit as st
from utils.document_processor import DocumentProcessor
from utils.chat_manager import ChatManager
import os

class RAGApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.chat_manager = ChatManager()
        self.initialize_session_state()
        
    @staticmethod
    def initialize_session_state():
        for key in ['processed', 'chat_history']:
            if key not in st.session_state:
                st.session_state[key] = [] if key == 'chat_history' else False
    
    def render_chat_interface(self):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Chat")
            for msg in st.session_state.chat_history:
                st.markdown(f"**You:** {msg['question']}")
                st.markdown(f"**Assistant:** {msg['response']}")
            
            question = st.text_area("Your question:", height=100)
            if st.button("Send") and question:
                response, context = self.chat_manager.get_response(question)
                st.session_state.chat_history.append({
                    "question": question,
                    "response": response,
                    "context": context
                })
                st.experimental_rerun()
        
        with col2:
            if st.session_state.chat_history:
                st.subheader("Context")
                st.markdown(st.session_state.chat_history[-1]["context"])
    
    def render_sidebar(self):
        with st.sidebar:
            st.header("Options")
            if st.button("Clear History"):
                st.session_state.chat_history = []
            if st.button("Reset System"):
                if os.path.exists("FAISS_Vectorstore.db"):
                    os.remove("FAISS_Vectorstore.db")
                st.session_state.processed = False
                st.session_state.chat_history = []
                st.experimental_rerun()
    
    def run(self):
        st.title("📚 RAG Chat System")
        
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
        if uploaded_file and not st.session_state.processed:
            with st.spinner("Processing..."):
                if self.doc_processor.process_pdf(uploaded_file):
                    st.session_state.processed = True
        
        if st.session_state.processed:
            self.render_chat_interface()
        
        self.render_sidebar()

if __name__ == "__main__":
    app = RAGApp()
    app.run()
#################################################################### WITHOUT STREAMLIT ########################################################################
import os
from dataclasses import dataclass
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from ollama import chat

@dataclass
class ChatMessage:
    question: str
    response: str
    context: str

class DocumentProcessor:
    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    def process_pdf(self, pdf_path: str) -> bool:
        try:
            documents = PyPDFLoader(pdf_path, extract_images=True).load()
            split_docs = self.text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            vectorstore.save_local("FAISS_Vectorstore.db")
            return True
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return False

class ChatManager:
    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.history: List[ChatMessage] = []

    def get_response(self, question: str) -> Tuple[str, str]:
        try:
            vectorstore = FAISS.load_local("FAISS_Vectorstore.db", self.embeddings, allow_dangerous_deserialization=True)
            relevant_docs = vectorstore.as_retriever().invoke(question)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            prompt = f"Context: {context}\n\nQuestion: {question}"
            response = chat(model="smollm:1.7b", messages=[{"role": "user", "content": prompt}])
            
            self.history.append(ChatMessage(question, response["message"]["content"], context))
            return response["message"]["content"], context
        except Exception as e:
            print(f"Error getting response: {e}")
            return "Error processing your question.", ""

    def display_history(self):
        for msg in self.history:
            print(f"\nQ: {msg.question}")
            print(f"A: {msg.response}")

class RAGSystem:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.chat_manager = ChatManager()

    def run(self):
        print("\n=== RAG Chat System ===")
        
        while True:
            print("\n1. Load PDF")
            print("2. Ask Question")
            print("3. View History")
            print("4. Exit")
            
            choice = input("\nSelect option (1-4): ")
            
            if choice == "1":
                pdf_path = input("Enter PDF path: ")
                if not os.path.exists(pdf_path):
                    print("File not found!")
                    continue
                    
                print("Processing PDF...")
                if self.doc_processor.process_pdf(pdf_path):
                    print("PDF processed successfully!")
                
            elif choice == "2":
                if not os.path.exists("FAISS_Vectorstore.db"):
                    print("Please load a PDF first!")
                    continue
                    
                question = input("\nEnter your question: ")
                print("\nGenerating response...")
                response, _ = self.chat_manager.get_response(question)
                print(f"\nResponse: {response}")
                
            elif choice == "3":
                self.chat_manager.display_history()
                
            elif choice == "4":
                print("Goodbye!")
                break
                
            else:
                print("Invalid option!")

if __name__ == "__main__":
    rag_system = RAGSystem()
    rag_system.run()
