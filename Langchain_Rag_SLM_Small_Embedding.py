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
