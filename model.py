import streamlit as st
from langchain_cohere import CohereEmbeddings
from langchain_openai import ChatOpenAI
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
OPENAI_API_KEY ="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"
COHERE_API_KEY ="VxKg2to3K9aZDtNzKUzjKIv6lpJelIp52VCjLFFq"

# Initialize embeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)

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
llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=OPENAI_API_KEY)

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["query", "context"],  # Ensure context is included
    template="""
You are an AI assistant providing accurate information about events, museums, and attractions in Saudi Arabia. 

### **Response Format**:
- **Event Name:** [Event Name]
- **Location:** [Location]
- **Description:** [Description]
- **Date:** [Event Start Date] - [Event End Date]
- **Ticket Price:** [Ticket Price]
- **Opening Hours:** [Opening Hours]

If no relevant data is found, respond with:
*"I'm sorry, but I don't have information about that event."*

### **User Query**:
{query}

### **Context (if available)**:
{context}
"""
)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template, "document_variable_name": "context"}
)

# Function to generate response
def generate_response(user_input: str) -> str:
    try:
        relevant_docs = retriever.get_relevant_documents(user_input)
        response = qa_chain.run({"query": user_input, "context": relevant_docs})
        return response if response else "I couldn't find any relevant information."
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
