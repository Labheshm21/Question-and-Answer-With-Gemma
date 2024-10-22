import os
import streamlit as st
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Fixed import
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load the Groq and Google Generative AI embeddings
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
st.title("Gemma Model Document Q&A")

# Initialize the ChatGroq LLM with the model name
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

# Define the prompt template for the LLM
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the questions:
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Initialize session state variables if not already done
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "loader" not in st.session_state:
    st.session_state.loader = None
if "docs" not in st.session_state:
    st.session_state.docs = None
if "final_documents" not in st.session_state:
    st.session_state.final_documents = None

# Define the function for vector embedding
def vector_embedding():
    if not st.session_state.vectors:
        # Initialize embeddings using Google Generative AI
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="model-embedding-001")  # Ensure the correct model name

        # Load documents from the specified directory
        st.session_state.loader = PyPDFDirectoryLoader("./us-cencus")  # Ensure folder is correctly named
        st.session_state.docs = st.session_state.loader.load()  # Load all documents

        # Check if documents are loaded
        if not st.session_state.docs:
            st.error("No documents were loaded.")
            return

        # Split documents into chunks for embedding
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        # Check if final_documents were generated
        if not st.session_state.final_documents:
            st.error("Documents were not split properly.")
            return

        # Debug: Check if embeddings are being generated for final_documents
        try:
            # Create FAISS vector store from the documents
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )
        except Exception as e:  # Catch all exceptions
            st.error(f"Error generating FAISS vectors: {e}")
            return

# User input for asking questions from the documents
prompt1 = st.text_input("What do you want to ask from the documents?")

# Button to trigger vector store creation
if st.button("Create Vector Store"):
    vector_embedding()
    st.write("Vector store DB is ready")

# Handling the user's query
if prompt1 and st.session_state.vectors:
    # Create document chain using the LLM and the prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Set up the retriever using the FAISS vectors
    retriever = st.session_state.vectors.as_retriever()

    # Create the retrieval chain using the retriever and document chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Record the start time and invoke the retrieval chain
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})

    # Display the answer
    st.write(response["answer"])

    # Show document similarity search details
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-------------------")
