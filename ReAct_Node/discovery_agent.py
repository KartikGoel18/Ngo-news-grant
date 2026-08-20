import json
from langchain.agents.middleware import ToolCallLimitMiddleware
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

load_dotenv()

from prompts import REACT_SYSTEM_PROMPT, config
from tools import react_tools

class ValidLink(BaseModel):
    title: str = Field(
        description="The explicit, clean title of the specific news article or grant webpage."
    )
    url: str = Field(
        description="The strictly valid, verified, non-blacklisted HTTPS URL of the resource."
    )
    type: Literal["news", "grant"] = Field(
        description="Categorization specifying whether the link represents a news event or a grant opportunity."
    )
    context_summary: str = Field(
        description="A concise one-sentence justification detailing why this link passed the quality/relevance filters."
    )
    publish_date: str = Field(
        description = "Publishing date for the article. If a date is not explicitly visible in the snippet, output 'Recent'."
    )

class DiscoveryOutput(BaseModel):
    valid_links: List[ValidLink] = Field(
        description="A verified, structurally validated list of highly relevant links for Indian NGOs. EXTREMELY IMPORTANT: You MUST extract and include ALL valid links found during the search. Do not truncate the list to a few items."
    )

    @field_validator("valid_links")
    @classmethod
    def filter_blacklisted_domains(cls, links: List[ValidLink]) -> List[ValidLink]:
        filtered = []
        for link in links:
            if not any(domain in link.url for domain in config.blacklisted_domains):
                filtered.append(link)
        return filtered

def create_impact_weaver_agent():
    # Creates and compiles the ReAct Agent Graph.
    limiter = ToolCallLimitMiddleware(
        run_limit=10, 
        exit_behavior="end" # Forces the agent to stop and emit a final message
    )

    # Initializing the LLM (via Groq)
    llm = ChatOpenAI(
        api_key=os.environ["LITELLM_API_KEY"],
        model="pro",
        base_url="https://llm.impactweaver.com",
        temperature=0.1,
    )
    
    # Build the Agent without response_format so it is FORCED to research and write out text
    agent_graph = create_agent(
        model=llm,
        tools=react_tools,
        system_prompt=REACT_SYSTEM_PROMPT,
        middleware=[limiter]
    )
    
    return agent_graph, llm

DEFAULT_MASTER_QUERY = (
    "Identify active grant applications, open calls for proposals, corporate CSR funding partnerships, "
    "and time-sensitive non-profit sector regulatory and current affairs news in India from the past week. Evaluate the "
    "relevance of these sources strictly by scanning the titles and text snippets returned by your search tool. "
    "Immediately discard informational guides, generic directories, overview articles, and 'how-to' explainers. "
    "Extract and compile a comprehensive list of all specific, actionable URLs found."
)

OUTPUT_FILE = Path(__file__).parent / "discovered_links.json"


def run_react_discovery(
    master_query: str = DEFAULT_MASTER_QUERY,
    output_path: Optional[Path] = OUTPUT_FILE,
) -> DiscoveryOutput:
    
    agent, llm = create_impact_weaver_agent()
    inputs = {"messages": [HumanMessage(content=master_query)]}
    response = agent.invoke(inputs)

    # 1. Native Token Tracking: Sum up tokens from the Agent's internal loop
    agent_input_tokens = 0
    agent_output_tokens = 0
    
    conversation_history = []
    for msg in response["messages"]:
        # Extract metadata if it exists on the message object
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            agent_input_tokens += msg.usage_metadata.get("input_tokens", 0)
            agent_output_tokens += msg.usage_metadata.get("output_tokens", 0)
            
        if isinstance(msg.content, str) and msg.content.strip():
            if "Tool call limit" in msg.content or "Do not make additional tool calls" in msg.content:
                continue
            conversation_history.append(msg.content)
            
    final_text = "\n\n--- NEXT STEP ---\n\n".join(conversation_history)
    print(f"\n[SYSTEM LOG] Cleaned context passed to parser: {len(final_text)} characters")

    pro_llm = ChatOpenAI(
        api_key=os.environ["LITELLM_API_KEY"],
        model="pro",
        base_url="https://llm.impactweaver.com/v1",
        temperature=0.1,
    )
    
    json_extraction_prompt = (
        "You are a precise JSON parsing assistant. Extract all the valid links, news, and grants "
        "from the following research summary into a strict JSON object matching this schema:\n\n"
        "{\n"
        '  "valid_links": [\n'
        "    {\n"
        '      "title": "String containing clean title",\n'
        '      "url": "String containing valid HTTPS URL",\n'
        '      "type": "news" or "grant",\n'
        '      "context_summary": "One-sentence justification",\n'
        '      "publish_date": "Date string or \'Recent\'"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "1. Respond ONLY with the valid JSON object. Do not wrap it in markdown block characters like ```json.\n"
        "2. Do not include introductory text or trailing notes.\n\n"
        f"Research Summary:\n{final_text}"
    )

    response_message = pro_llm.invoke(json_extraction_prompt)
    raw_response = response_message.content
    
    # 2. Native Token Tracking: Get tokens from the JSON parser's single output
    parser_usage = response_message.usage_metadata or {}
    parser_input_tokens = parser_usage.get("input_tokens", 0)
    parser_output_tokens = parser_usage.get("output_tokens", 0)
    
    cleaned_json = raw_response.strip()
    if cleaned_json.startswith("```json"):
        cleaned_json = cleaned_json[7:]
    if cleaned_json.startswith("```"):
        cleaned_json = cleaned_json[3:]
    if cleaned_json.endswith("```"):
        cleaned_json = cleaned_json[:-3]
    cleaned_json = cleaned_json.strip()

    try:
        structured_response = DiscoveryOutput.model_validate_json(cleaned_json)
    except Exception as validation_error:
        print(f"\n[!] JSON Validation Error: {validation_error}")
        print("--- Raw output received from model ---")
        print(raw_response)
        print("--------------------------------------")
        return None

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structured_response.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"[+] SUCCESS: Discovered links successfully written to {output_path}")
    else:
        print("[+] SUCCESS: Discovered links generated in memory")

    # 3. Combine and print the grand totals!
    total_input = agent_input_tokens + parser_input_tokens
    total_output = agent_output_tokens + parser_output_tokens
    
    print("\n================ TOTAL SCRIPT TOKEN USAGE ================")
    print(f"Total Tokens Consumed:      {total_input + total_output}")
    print(f"Prompt (Input) Tokens:      {total_input}")
    print(f"Completion (Output) Tokens: {total_output}")
    print("==========================================================\n")

    return structured_response


if __name__ == "__main__":
    run_react_discovery()
