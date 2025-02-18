import streamlit as st
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
from langchain_core.documents import Document
from langchain_community.llms import Cohere
from langchain.chains import RetrievalQA
import time

# Set up the Streamlit app title
st.title("RAG Chatbot for Event Information")

# Step 2: Set up the Cohere API key (use Streamlit secrets)
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]

# Step 3: Initialize Cohere embeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)

# Step 4: Load and preprocess the data
@st.cache_data  # Cache the data to avoid reloading on every interaction
def load_data():
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

data = load_data()

# Step 5: Convert text data into Document objects
documents = [Document(page_content=text) for text in data['combined_text']]

# Step 6: Create a vector store
# Create a FAISS vector store from the documents (with rate limiting)
batch_size = 10  # Adjust based on your rate limits
vector_store = None

for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    if vector_store is None:
        vector_store = FAISS.from_documents(batch, embeddings)
    else:
        vector_store.add_documents(batch)
    time.sleep(1)  # Add a 1-second delay between batches


# Step 7: Create a retriever
retriever = vector_store.as_retriever()

# Step 8: Initialize the Cohere language model
llm = Cohere(model="command-r-plus")

# Step 9: Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Step 10: Define a function to generate responses
def generate_response(user_input: str) -> str:
    return qa_chain.run(user_input)

# Streamlit UI
st.write("Welcome to Aoun Bot! Ask me anything about events.")

# Input box for user query
user_input = st.text_input("Enter your question:")

# Generate and display response
if user_input:
    with st.spinner("Generating response..."):  # Show a spinner while processing
        response = generate_response(user_input)
    st.write("**Response:**")
    st.write(response)