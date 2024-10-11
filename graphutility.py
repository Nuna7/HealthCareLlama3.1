from models import embedding_tokenizer, embedding_model
from utility import generate_answer, grade_document
from memory import memory,process_docs, collection
from search import web_search_tool
import torch
from langchain.schema import Document

def retrieve(state):
    """
    Retrieve relevant text for a question.
    CPU is used due to limited GPU memory.
    """
    question = state["question"]
    
    inputs = embedding_tokenizer(question, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, max_length=512).to("cpu")
    
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()  
    
    documents = collection.query(
        query_embeddings=embeddings.tolist(),
        n_results=3
    )
    
    state['documents'] = documents
    
    return state

def generate(state):
    """
    Generate the answer.
    """

    answer, update_state = generate_answer(state)
    state['generation'] = answer
    state['conversation_history'] = update_state['conversation_history']
    return state

def grade_documents(state):
    """
    Grade the document on whether it is relevant for the question or not.
    """
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    search = "Yes"
    for d in documents:
        score = grade_document(question, d)
        
        if score == "yes":
            filtered_docs.append(d)
            search = "No"
    state['documents'] = filtered_docs
    state['search'] = search
    
    return state

def web_search(state):
    """
    Search new relevant information for the given question.
    """
    question = state["question"]
    documents = state.get("documents", [])
    web_results = web_search_tool.invoke({"query": question})
    
    new_documents = [
        Document(page_content=d["content"], metadata={"source": d["url"]})
        for d in web_results
    ]
    
    processed_docs = process_docs(new_documents, collection)
    documents.extend(processed_docs)
    
    state['documents'] = documents
    
    return state

def decide_to_generate(state):
    """
    Decide whether to search or generate.
    """
    search = state["search"]
    if search == "Yes":
        return "search"
    else:
        return "generate"