import streamlit as st
from langchain_huggingface import HuggingFacePipeline
from models import generation_pipeline, embedding_tokenizer, embedding_model, text_splitter
from langchain.schema import Document
import chromadb
import pdfplumber
import torch
import gc
import numpy as np

from langgraph.checkpoint.memory import MemorySaver

@st.cache_resource
def initialize_memory():
    """
    Stored the conversation between user and the system.
    """
    memory = MemorySaver()
    return memory

@st.cache_resource
def load_chroma():
    """
    Vector database for storing crucial information.
    """
    persist_directory = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    return chroma_client


def get_embeddings(text, embedding_tokenizer, embedding_model):
    """
    Yield the embedding of an input text
    """
    inputs = embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cpu")
    
    all_embeddings = []
    
    for i in range(0, inputs['input_ids'].size(1), 512):
        input_chunk = {k: v[:, i:i+512] for k, v in inputs.items()}
        with torch.no_grad():
            outputs = embedding_model(**input_chunk)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(embeddings)
        
    gc.collect()
    torch.cuda.empty_cache()
    
    aggregated_embeddings = np.mean(np.array(all_embeddings), axis=0)
    return aggregated_embeddings
    

def check_collection(chroma_client):
    """
    Check whether vector database exist or not. If not created one.
    """
    existing_collections = chroma_client.list_collections()
    collection_name = "health_information"
    collection_exists = any(col.name == collection_name for col in existing_collections)

    if collection_exists:
        collection = chroma_client.get_collection(name=collection_name)
    else:
        collection = chroma_client.create_collection(name="health_information")
        initial_file_path = "./doc1.pdf"
        with pdfplumber.open(initial_file_path) as pdf:
            docs = []
            for page in pdf.pages:
                text = page.extract_text()
                docs.append(text)

        docs_list = [Document(page_content=doc) for doc in docs]
        doc_splits = text_splitter.split_documents(docs_list)

        embeddings = [get_embeddings(doc.page_content, embedding_tokenizer, embedding_model) for doc in docs_list]

        for i, embedding in enumerate(embeddings):
            flat_embedding = embedding.flatten().tolist()
            collection.add(
                embeddings=[flat_embedding],
                metadatas=[{"content": docs_list[i].page_content}],
                ids=[f"doc_{i}"]
            )
            
    return collection

def process_docs(documents, collection):
    """
    Output an embedding for documents.
    """
    doc_splits = text_splitter.split_documents(documents)
    embeddings = [get_embeddings(doc.page_content, embedding_tokenizer, embedding_model) for doc in doc_splits]
    
    embeddings_flat = [embedding.flatten().tolist() for embedding in embeddings]
    metadatas = [{"content": doc.page_content} for doc in doc_splits]
    ids = [f"doc_{i}" for i in range(len(doc_splits))]
    
    collection.add(
        embeddings=embeddings_flat,
        metadatas=metadatas,
        ids=ids
    )
    
    gc.collect()
    torch.cuda.empty_cache()
        
    return doc_splits


memory = initialize_memory()
chroma_client = load_chroma()
collection = check_collection(chroma_client)  