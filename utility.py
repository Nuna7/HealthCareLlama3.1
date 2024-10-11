import gc
import numpy as np
import torch
from prompts import prompt, grading_prompt
from models import text_splitter, embedding_tokenizer, embedding_model, grading_pipeline, generation_pipeline
from postprocessing import clean_response

def generate_answer(state):
    """
    Generate answer given chat_history, question and context from user input and vector database
    """
    question = state['question']
    documents = state['documents']
    context = state['context']
    chat_history = state.get("conversation_history", [])
    
    if isinstance(question, torch.Tensor):
        question = question.item() if question.numel() == 1 else question.tolist()
    question = str(question)
    
    if chat_history == []:
        history = ""
        state['conversation_history'] = []
    else:
        history = '\n'.join(chat_history)
        
    
    formatted_prompt = prompt.format(question=question, documents=documents, chat_history=history, context=context)
    
    tokenized_input = generation_pipeline.tokenizer(
        formatted_prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=generation_pipeline.model.config.max_position_embeddings  
    )

    input_length = tokenized_input['input_ids'].shape[-1]

    max_new_tokens = generation_pipeline.model.config.max_position_embeddings - input_length
    max_new_tokens = max(1, max_new_tokens)

    with torch.no_grad():
        response = generation_pipeline(
            formatted_prompt,
            max_length=input_length + max_new_tokens,  
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
        
    answer = clean_response(response[0]["generated_text"])
    
    state['conversation_history'].append(f"User: {question}")
    state['conversation_history'].append(f"AI: {answer}")
    
    if len(state['conversation_history']) > 3:
        state['conversation_history'] = state['conversation_history'][-3:]
        
    gc.collect()
    torch.cuda.empty_cache()
    
    return answer, state

def grade_document(question, document):
    """
    Grade whether the document are relevant for the question or not.
    """
    formatted_prompt = grading_prompt.format(question=question, document=document)
    
    with torch.no_grad():
        response = grading_pipeline(formatted_prompt)[0]
    gc.collect()
    torch.cuda.empty_cache()

    if response[0]['label'] == 'LABEL_1':
        return "yes"
    else:
        return "no"

    
def load_device():
    """
    Load Device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
