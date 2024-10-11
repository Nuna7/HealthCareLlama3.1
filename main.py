import streamlit as st
import io

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import pdfplumber

import torch
import os
import re

from langchain_core.output_parsers import StrOutputParser
import uuid

import pdf2image
import pytesseract

from postprocessing import clean_response
from graph_ import custom_graph

from memory import get_embeddings, collection, process_docs
from langchain.schema import Document
from models import text_splitter, embedding_tokenizer, embedding_model

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

output_parser = StrOutputParser()

def predict_custom_agent_answer(user_input: dict):
    """
    Running the agent.
    """
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    with st.spinner('Processing your question...'):
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
            
        initial_state = {
            "question": user_input["input"],
            "context": user_input['context'],
            "conversation_history": st.session_state.conversation_history
        }
        
        state_dict = custom_graph.invoke(initial_state, config)
        
        st.session_state.conversation_history = state_dict['conversation_history']
        
        parsed_response = output_parser.parse(state_dict['generation'])
        return parsed_response

def main():
    """
    Main app.
    """
    st.title('Online Doctor')
    st.write("This application helps to answer your health-related questions using relevant documents.")

    page = st.sidebar.selectbox("Choose a function", ["Ask a Question", "Upload PDF"])

    if page == "Ask a Question":
        ask_question_page()
    elif page == "Upload PDF":
        upload_pdf_page()

def ask_question_page():
    """
    Asking question page.
    """
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    question = st.text_input("Enter your health-related question:")
    pdf_file = st.file_uploader("Upload any personal PDF file insignificant for the question", type=["pdf"])

    if st.button("Get Answer"):
        if pdf_file is not None:
            images = pdf2image.convert_from_bytes(pdf_file.read(), dpi=300)
            extracted_text = []
            for image in images:
                text = pytesseract.image_to_string(image)
                extracted_text.append(text)
                
            full_text = "\n".join(extracted_text)
            cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
        else:
            cleaned_text = ""

        user_input = {"input": question, "context": cleaned_text}
        
        try:
            response = predict_custom_agent_answer(user_input)
        except Exception as e:
            response = "I'm sorry, but I encountered an error while generating your answer. Please try again with a different query."
    
    if len(st.session_state.conversation_history) > 0:
        st.write("**Conversation History:**")
        for message in st.session_state.conversation_history:
            message = message.split(":")
            speaker = message[0]
            message = message[1]
            st.write(f"**{speaker}:** {message}")

def upload_pdf_page():
    """
    Upload pdf page.
    """
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner('Processing your PDF...'):
            uploaded_pdf = io.BytesIO(uploaded_file.read())
            with pdfplumber.open(uploaded_pdf) as pdf:
                docs = []
                for page in pdf.pages:
                    text = page.extract_text()
                    docs.append(text)
                    
            docs_list = [Document(page_content=doc) for doc in docs]
            process_docs(docs_list, collection)

            st.write("PDF uploaded and processed successfully!")

if __name__ == "__main__":
    main()