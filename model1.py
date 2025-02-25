import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_community.vectorstores import FAISS
import pandas as pd
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
import time
from langchain.prompts import PromptTemplate
import os


# Set up the Streamlit app title
st.set_page_config(page_title="Aoun Bot", page_icon="🤖", layout="centered")
st.title("🤖 Aoun Bot ")
st.write("Ask about events, museums, or attractions in Saudi Arabia!")

# Step 2: Set up the OpenAI API key
OPENAI_API_KEY = "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"


# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load and preprocess the data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('Gp-data (PROTOTYPE).csv', encoding='latin-1')
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

# Convert text data into Document objects
documents = [Document(page_content=text) for text in data['combined_text']]

# Create FAISS vector store
@st.cache_resource
def create_vector_store(_docs):
    vector_store = FAISS.from_documents(_docs[:10], embeddings)
    batch_size = 10
    for i in range(10, len(_docs), batch_size):
        try:
            vector_store.add_documents(_docs[i:i + batch_size])
            time.sleep(1)
        except Exception as e:
            st.error(f"Error adding documents: {e}")
            break
    return vector_store

vector_store = create_vector_store(documents)
if not vector_store:
    st.error("Failed to create the vector store.")
    st.stop()

retriever = vector_store.as_retriever()

# Initialize OpenAI language model
llm = ChatVertexAI(model_name="gemini-pro")

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
    input_key="query",  # Explicitly specify the input key
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