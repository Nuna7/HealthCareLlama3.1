from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os

from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv('HF_TOKEN')
os.environ['HF_TOKEN'] = hf_token

@st.cache_resource
def load_model():
    """
    Load llama3.1 for question answer. Use 8bit quantization for memory efficiency.
    """
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True  
    )
    generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer,
                                  device_map="auto")
    return  generation_pipeline

@st.cache_resource
def load_embedding_model():
    """
    Load BioBert model to represent the text information stored, retreive and even compare.
    CPU is used due to limited memory.
    """
    embedding_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    embedding_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to("cpu")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000,
                                                                         chunk_overlap=25)
    
    return embedding_tokenizer, embedding_model, text_splitter

@st.cache_resource
def load_grading_model():
    """
    Load DistilBert model to classify between relevant and irrelevant text which decide whether new informations are to be searched or not.
    CPU is used due to limited memory.
    """
    grading_model_name = "distilbert-base-uncased-finetuned-sst-2-english" 
    grading_model = AutoModelForSequenceClassification.from_pretrained(grading_model_name, num_labels=2)
    grading_tokenizer = AutoTokenizer.from_pretrained(grading_model_name)
    grading_pipeline = pipeline("text-classification",model=grading_model,tokenizer=grading_tokenizer,top_k=1,device="cpu")
    return grading_pipeline


generation_pipeline = load_model()
embedding_tokenizer, embedding_model, text_splitter = load_embedding_model()    
grading_pipeline = load_grading_model()