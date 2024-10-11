from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict, List
from graphutility import retrieve, generate, grade_documents, web_search, decide_to_generate
from memory import memory

class GraphState(TypedDict):
    """
    The state object for the entire conversation
    """
    question: str
    context: str
    generation: str
    search: str
    documents: List[str]
    conversation_history : List[str]
    
def initialise_workflow():
    """
    Initialising the state object.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)  
    workflow.add_node("grade_documents", grade_documents)  
    workflow.add_node("generate", generate) 
    workflow.add_node("web_search", web_search)  

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    custom_graph = workflow.compile(checkpointer=memory)
    return custom_graph

custom_graph = initialise_workflow()