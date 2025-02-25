import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
import pandas as pd
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
import time
from langchain.prompts import PromptTemplate
import os


# Set up Streamlit app
st.set_page_config(page_title="Aoun Bot", page_icon="🤖", layout="centered")
st.title("🤖 Aoun Bot")
st.write("Ask about events, museums, or attractions in Saudi Arabia!")

# Load API Keys securely from Streamlit Secrets
OPENAI_API_KEY = "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"


# Initialize OpenAI language model
llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=OPENAI_API_KEY)

# Initialize Hugging Face Embeddings (Free & Local)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# File paths
DATA_PATH = "Gp-data (PROTOTYPE).csv"
FAISS_INDEX_PATH = "faiss_index"

# Load and preprocess dataset
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found: {DATA_PATH}. Upload it to the GitHub repo.")
        st.stop()

    try:
        data = pd.read_csv(DATA_PATH, encoding='latin-1')
        data['combined_text'] = (
            "Description: " + data['Description'].astype(str) + ". " +
            "Location: " + data['Location'].astype(str) + ". " +
            "Ages: " + data['Ages'].astype(str) + ". " +
            "Ticket Price: " + data['Ticket Price'].astype(str) + ". " +
            "Event Start Date: " + data['Event Start Date'].astype(str) + ". " +
            "Event End Date: " + data['Event End Date'].astype(str) + ". " +
            "Opening Hours: " + data['Opening Hours'].astype(str) + "."
        )
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

data = load_data()
if data.empty:
    st.error("No data available. Please check the dataset and try again.")
    st.stop()

# Convert dataset into LangChain Document objects
documents = [Document(page_content=text) for text in data['combined_text']]

# Create or Load FAISS Vector Store
@st.cache_resource
def create_or_load_vector_store(_docs):
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
            return vector_store
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")
    
    vector_store = FAISS.from_documents(_docs, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store

vector_store = create_or_load_vector_store(documents)
if not vector_store:
    st.error("Failed to create or load the vector store.")
    st.stop()

# Set up retriever
retriever = vector_store.as_retriever()

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template="""
You are an AI assistant providing accurate information about events, museums, and attractions in Saudi Arabia. 

### **Response Format**:
- **Event Name:** [place]
- **Location:** [Location]
- **Description:** [Description]
- **Date:** [Event Start Date] - [Event End Date]
- **Ticket Price:** [Ticket Price]
- **Opening Hours:** [Opening Hours]

If no relevant data is found, respond with:
*"I'm sorry, but I don't have information about that event."*

### **User Query**:
{query}

### **Context**:
{context}
"""
)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# Function to generate response
def generate_response(user_input: str) -> str:
    try:
        # Debugging: Print the input dictionary
        input_dict = {"query": user_input}
        print("Input Dictionary:", input_dict)  # Debugging line

        # Pass the input dictionary to the chain
        response = qa_chain.invoke(input_dict)
        if "result" in response:
            return response["result"]
        else:
            return "I couldn't find any relevant information."
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "An error occurred. Please try again."

# User input box
user_input = st.text_input("Enter your question:")

# Generate and display response
if user_input:
    with st.spinner("Generating response..."):
        response = generate_response(user_input)
    st.write("**Response:**")
    st.write(response)
