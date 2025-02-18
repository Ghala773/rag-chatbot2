# import necessary libraries
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnablePassthrough
import os
from sklearn.metrics import precision_score, recall_score


# 1. Document Processing
# Load and process dataset using CSVLoader
file_path = "Gp-data (PROTOTYPE).csv"
loader = CSVLoader(file_path=file_path)
data = loader.load()
# Combine relevant columns into a single searchable text field
# This helps in creating meaningful embeddings that capture multiple attributes of the museum
text_data = [Document(page_content=doc.page_content) for doc in data]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(text_data)
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(chunks, embedding_model)
def search(query, k=3, threshold=0.70):
    query_embedding = embedding_model.embed_query(query)
    docs_and_scores = vector_store.similarity_search_with_score(query, k=k)

    # Filter results based on cosine similarity threshold to ensure high relevance
    filtered_results = [(doc, score) for doc, score in docs_and_scores if score >= threshold]

    return filtered_results
query = "What are the events in Riyadh?"
results = search(query)
print("Search Results:")
for i, (doc, score) in enumerate(results):
    print(f"{i+1}. {doc.page_content} (Similarity Score: {score})")
    llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key="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")
prompt_template = PromptTemplate(
    template="""
You are an AI assistant specialized in providing information about events in Saudi Arabia.
Your task is to answer user queries accurately based on the available dataset. Follow these strict rules:

### **General Rules**:
1. **Always provide complete and structured answers** with details about the event name, city, description, date, and ticket price.
2. **If an event has a specific location, always include it** in the response.
3. **Do not make up any information**; only use data that exists in the dataset.
4. **If the user asks about events in a specific city**, filter results to match that city.
5. **If the user asks about a specific event type**, respond with only those matching the specified type.
6. **If no relevant data is found**, respond with:
   *"I apologize, I don't have information about that event."*
7. **Use formal and structured responses** in clear English.

### **Response Format**:
- **Event Name:** [Event Name]
- **City:** [City Name]
- **Description:** [Event Description]
- **Date:** [Event Start Date] - [Event End Date]
- **Ticket Price:** [Ticket Price]
- **Opening Hours:** [Opening Hours]

### **Additional Context Handling**:
- If the user asks for **all events in a city**, list all that match.
- If the user asks for **concerts, exhibitions, or sports events**, provide only those relevant.
- If the event has an **age restriction**, clearly mention it.

### **User Query & Context**:
Query: {query}
Context: {context}

### **Final Answer**:
""",
    input_variables=["query", "context"]
)

rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever()
)

query = "What are the events in Riyadh??"
answer = rag_chain.invoke({"question": query, "chat_history": []})

def display_results(answer):
    print("\nGenerated Answer:\n")

    if isinstance(answer, dict):  # Check if the answer is a dictionary
        for key, value in answer.items():
            print(f"**{key}:**\n{value}\n")   # Print each key and its value on a new line
    else:
        print(answer)  # If it's not a dictionary, print it directly

display_results(answer)

