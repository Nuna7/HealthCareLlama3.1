from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""You are a health assistant for question-answering tasks.

    Use the following information to answer the question.

    If you don't know the answer, just say that you don't know.

    Keep the answer concise and relevant to the current question.
    
    Answer in second person.

    Chat History:
    {chat_history}
    
    Context:
    {context}

    Current Question: {question}
    Relevant information: {documents}
    Answer:
    """,
    input_variables=["question", "documents", "chat_history","context"],
)

grading_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    Output only the binary score.
    """,
    input_variables=["question", "document"],
)
