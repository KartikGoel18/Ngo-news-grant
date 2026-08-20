from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

# DDG search specifically for the past week ('w') and in India ('in-en')
ddg_wrapper = DuckDuckGoSearchAPIWrapper(region="in-en", time="w", max_results=10)
ddg_search = DuckDuckGoSearchResults(api_wrapper=ddg_wrapper, output_format="list")

@tool
def search_latest_ngo_info(optimized_query: str) -> str:
    """
    Search the internet for the latest news and grants based on the query.
    This tool automatically restricts results to the past week and focuses on Indian context.
    
    Args:
        optimized_query (str): The specific, cleaned search string you planned.
        
    Returns:
        A string containing a list of search results including titles, snippets, and URLs.
    """
    print(f"\n[SYSTEM LOG] Executing Live Search for: '{optimized_query}'")
    
    try:
        # Get raw results as a list of dicts
        raw_results = ddg_search.run(optimized_query) 
            
        cleaned_results = []
        # Blacklist words in URLs that indicate static/junk pages
        blacklist = ['tracxn.com', 'exam-prep', 'wikipedia', 'amazon', 'blog', 'advertisement']
            
        for result in raw_results:
            url = result.get('link', '').lower()
            # Python-level structural filtering
            if not any(bad_word in url for bad_word in blacklist):
                cleaned_results.append(result)
                    
        return cleaned_results
    except Exception as e:
        return f"Search failed with error: {str(e)}"

react_tools = [search_latest_ngo_info]