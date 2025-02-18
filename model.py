import streamlit as st
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
from langchain_core.documents import Document
from langchain_community.llms import Cohere
from langchain.chains import RetrievalQA
import time

# Set up the Streamlit app title
st.set_page_config(page_title="Aoun Bot", page_icon="🤖", layout="centered")
st.title("🤖 Aoun Bot ")
st.write("Ask me about events happening in Saudi Arabia!")
# Step 2: Set up the Cohere API key (use Streamlit secrets)
COHERE_API_KEY = "O2vtG9p7cz5fXf77D7BYkvbJq3depsVWQGbCFJnR"

# Step 3: Initialize Cohere embeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)

# Step 4: Load and preprocess the data
@st.cache_data  # Cache to improve performance
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
    st.error("No data available. Please check the CSV file.")
    st.stop()

# Step 5: Convert text data into Document objects
documents = [Document(page_content=text) for text in data['combined_text']]

# Step 6: Create a FAISS vector store
@st.cache_resource
def create_vector_store(_docs):  # Underscore prevents hashing issues
    vector_store = FAISS.from_documents(_docs[:10], embeddings)  # Initialize with first batch
    batch_size = 10  # Process in small batches
    for i in range(10, len(_docs), batch_size):
        vector_store.add_documents(_docs[i:i + batch_size])
        time.sleep(1)  # Delay to prevent API rate limit issues
    return vector_store

vector_store = create_vector_store(documents)
retriever = vector_store.as_retriever()

# Step 7: Initialize Cohere language model
llm = Cohere(model="command-r-plus", cohere_api_key=COHERE_API_KEY)

# Step 8: Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Function to generate response
def generate_response(user_input: str) -> str:
    try:
        return qa_chain.run(user_input)
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit UI
st.write("### Ask Aoun Bot about events!")

# Input box for user query
user_input = st.text_input("Enter your question:")

# Generate and display response
if user_input:
    with st.spinner("Generating response..."):  # Show a spinner while processing
        response = generate_response(user_input)
    st.write("**Response:**")
    st.write(response)