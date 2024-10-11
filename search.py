from langchain_community.tools.tavily_search import TavilySearchResults
import os

from dotenv import load_dotenv

load_dotenv()

tavily_api_key = os.getenv('TAVILY_API_KEY')
os.environ['TAVILY_API_KEY'] = tavily_api_key

@st.cache_resource
def load_tavily():
    """
    Searching tool for new information.
    """
    web_search_tool = TavilySearchResults()
    return web_search_tool

web_search_tool = load_tavily()