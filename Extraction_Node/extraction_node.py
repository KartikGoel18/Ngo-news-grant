
"""
extraction_node.py
==================
Extract structured JSON data from a webpage using Crawl4AI's LLM-based
extraction strategy (`LLMExtractionStrategy`).

This file follows ONLY what is documented at:
    https://docs.crawl4ai.com/extraction/llm-strategies/

No extra Crawl4AI features (deep crawling, CSS extraction, hooks, etc.)
are used — just the plain LLM extraction flow described in the docs:

    1. Define a Pydantic model  -> becomes the JSON `schema`
    2. Build an `LLMExtractionStrategy` with that schema
    3. Put the strategy inside a `CrawlerRunConfig`
    4. Run `AsyncWebCrawler.arun(url, config=...)`
    5. Read `result.extracted_content` (a JSON string) and `json.loads()` it
    6. Optionally print token usage with `llm_strategy.show_usage()`

Before running:
    pip install -r requirements.txt
    crawl4ai-setup                     # installs/configures the headless browser
    $env:LITELLM_API_KEY="..."         # or put it in the project .env file
    $env:LITELLM_BASE_URL="https://llm.impactweaver.com"

Usage:
    python Extraction_Node/extraction_node.py "https://example.com/article" \
        --type news \
        --provider "litellm_proxy/lite"
"""

import os
import json
import asyncio
import argparse
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from crawl4ai import (
        AsyncWebCrawler,
        BrowserConfig,
        CrawlerRunConfig,
        CacheMode,
        LLMConfig,
        LLMExtractionStrategy,
    )
    try:
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    except ImportError:
        DefaultMarkdownGenerator = None
    _CRAWL4AI_IMPORT_ERROR = None
except ImportError as e:
    AsyncWebCrawler = None
    BrowserConfig = None
    CrawlerRunConfig = None
    CacheMode = None
    LLMConfig = None
    LLMExtractionStrategy = None
    DefaultMarkdownGenerator = None
    _CRAWL4AI_IMPORT_ERROR = e


# ---------------------------------------------------------------------------
# 1. DEFINE THE OUTPUT SCHEMA
# ---------------------------------------------------------------------------
# Crawl4AI's LLMExtractionStrategy can extract data conforming to a JSON
# schema. The docs recommend building that schema from a Pydantic model via
# `YourModel.model_json_schema()`.
#
# Grants and news items don't actually share one shape — a grant needs
# funding-specific fields (amount, eligibility, application deadline/URL)
# that make no sense on a news item, and a news item needs fields (publisher,
# published date) that make no sense on a grant. Rather than one flat model
# with a pile of "fill this in only if it applies" optional fields, this
# defines two separate Pydantic models — `GrantItem` and `NewsItem` — and
# combines them into a *discriminated union* keyed on `category`. Pydantic
# (v2) turns that into a JSON schema with a `oneOf` + `discriminator`, so the
# schema itself documents which fields belong to which category instead of
# relying on the LLM to guess which optional fields are relevant.
#
# NOTE: this is still the exact same documented mechanism — a Pydantic
# model's `model_json_schema()` passed into `LLMExtractionStrategy` — just a
# richer model. If your chosen LLM provider struggles with `oneOf`/
# `discriminator` schemas (some smaller/local models handle nested schemas
# less reliably), a simple fallback is to run `extract_from_url()` twice:
# once with a `GrantItem`-only schema/instruction, once with a
# `NewsItem`-only schema/instruction, then merge the two result lists.
# ---------------------------------------------------------------------------
class GrantItem(BaseModel):
    """Schema used when an extracted item is a grant / funding scheme."""
    category: Literal["grant"] = Field(
        "grant", description="Fixed discriminator value for grant items"
    )
    title: str = Field(..., description="The name of the grant or funding scheme")
    description: Optional[str] = Field(
        None, description="A short summary of what the grant funds or its purpose"
    )
    funding_amount: Optional[str] = Field(
        None,
        description="The grant amount, funding size, or value range, if stated",
    )
    eligibility: Optional[str] = Field(
        None, description="Who is eligible to apply for this grant, if stated"
    )
    start_date: Optional[str] = Field(
        None, description="The date applications open, if present"
    )
    end_date: Optional[str] = Field(
        None,
        description="The application deadline / closing date for the grant, if present",
    )
    application_mode: Optional[str] = Field(
        None,
        description="Mode of application, such as online, offline, email, postal, or not specified",
    )
    application_url: Optional[str] = Field(
        None,
        description="The URL for applying, but only if the application mode is online and a link is present",
    )
    main_media_url: Optional[str] = Field(
        None, description="Link for the main media file (image or video) if it exists, otherwise null"
    )


class NewsItem(BaseModel):
    """Schema used when an extracted item is a news item / announcement."""
    category: Literal["news"] = Field(
        "news", description="Fixed discriminator value for news items"
    )
    title: str = Field(..., description="The headline of the news item")
    summary: Optional[str] = Field(
        None,
        description=(
            "A complete summary of all important points from the news item, "
            "including key facts, stakeholders, dates, policy or funding "
            "implications, and why it matters."
        ),
    )
    published_date: Optional[str] = Field(
        None, description="The date the news item was published or posted, if present"
    )
    publisher: Optional[str] = Field(
        None,
        description="The publisher, department, or author of the news item, if stated",
    )
    url: Optional[str] = Field(
        None, description="A link to the full news article, if present"
    )
    main_media_url: Optional[str] = Field(
        None, description="Link for the main media file (image or video) if it exists, otherwise null"
    )


# The discriminated union: Pydantic selects GrantItem vs. NewsItem based on
# the value of the `category` field, and the generated schema documents both
# shapes so the LLM knows exactly which fields go with which category.
ExtractedItem = Annotated[Union[GrantItem, NewsItem], Field(discriminator="category")]


class ExtractionResult(BaseModel):
    """Top-level container: the list of items the LLM extracts from the page."""
    items: List[ExtractedItem] = Field(
        default_factory=list,
        description="All items extracted from the page content, each either a grant or a news item",
    )


class GrantExtractionResult(BaseModel):
    """Type-specific container used when the upstream discovery node says this URL is a grant."""
    items: List[GrantItem] = Field(
        default_factory=list,
        description=(
            "The grant or funding opportunity record extracted from the page. "
            "Return exactly one item for this URL."
        ),
    )


class NewsExtractionResult(BaseModel):
    """Type-specific container used when the upstream discovery node says this URL is news."""
    items: List[NewsItem] = Field(
        default_factory=list,
        description=(
            "The news record extracted from the page. Return exactly one item "
            "for this URL."
        ),
    )


OUTPUT_FILE = Path(__file__).parent / "extracted_items.json"

DEFAULT_EXTRACTION_INSTRUCTION = (
    "Extract all relevant items from the page content as structured JSON. "
    "For each item, first decide whether it is a 'grant' (a funding scheme, "
    "scholarship, or application opportunity) or a 'news' item (an announcement, "
    "update, or press release). Then fill in only the fields that apply to that category."
)

GRANT_EXTRACTION_INSTRUCTION = (
    "Extract grant, funding, CSR partnership, open-call, scholarship, or application "
    "opportunity details from the main page content only. Return exactly one grant "
    "item for this URL. Include the title, description, funding_amount, eligibility, "
    "start_date, end_date or deadline, application_mode, and application_url whenever "
    "they are present. IMPORTANT: Extract the URL of the first available relevant image "
    "(e.g., from ![alt](url) or <img src='url'>) into main_media_url. Even if it's not "
    "strictly a 'main' image, pick the first image in the article! "
    "Ignore ads, generic guides, and unrelated recommendations."
)

NEWS_EXTRACTION_INSTRUCTION = (
    "Extract news, regulatory updates, current affairs, announcements, or press-release "
    "content from the main article/page content only. Return exactly one news item "
    "for this URL. Put the headline/title of the article in 'title'. Put all "
    "important points into the 'summary' key as one paragraph: key facts, dates, "
    "stakeholders, policy or funding implications, sector impact, and why it matters "
    "for Indian NGOs or the social-impact sector. Put the posted date in "
    "'published_date'. Put the publisher, department, or author in 'publisher'. "
    "Put the original article URL in 'url'. IMPORTANT: Extract the URL of the first "
    "available relevant image (e.g., from ![alt](url) or <img src='url'>) into "
    "main_media_url. Even if it's not strictly a 'main' image, pick the first image "
    "in the article! Ignore ads, comments, sidebars, and unrelated recommendations."
)


def _normalise_link_type(link_type: Optional[str]) -> Optional[Literal["grant", "news"]]:
    if link_type is None:
        return None
    normalised = str(link_type).strip().lower()
    if normalised in {"grant", "grants", "funding", "funding_opportunity", "opportunity"}:
        return "grant"
    if normalised in {"news", "article", "announcement", "press_release", "current_affairs"}:
        return "news"
    raise ValueError(f"Unsupported link type: {link_type!r}. Expected 'grant' or 'news'.")


def _instruction_for_type(link_type: Optional[str]) -> str:
    normalised = _normalise_link_type(link_type)
    if normalised == "grant":
        return GRANT_EXTRACTION_INSTRUCTION
    if normalised == "news":
        return NEWS_EXTRACTION_INSTRUCTION
    return DEFAULT_EXTRACTION_INSTRUCTION


def _schema_for_type(link_type: Optional[str]) -> dict:
    normalised = _normalise_link_type(link_type)
    if normalised == "grant":
        return GrantExtractionResult.model_json_schema()
    if normalised == "news":
        return NewsExtractionResult.model_json_schema()
    return ExtractionResult.model_json_schema()


def _api_token_for_provider(provider: str) -> Optional[str]:
    provider_name = provider.split("/", 1)[0].lower()
    if provider_name == "litellm_proxy":
        return os.getenv("LITELLM_API_KEY")
    if provider_name == "groq":
        return os.getenv("GROQ_API_KEY")
    if provider_name == "openai":
        return os.getenv("OPENAI_API_KEY")
    if provider_name == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    return os.getenv("LITELLM_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")


def _base_url_for_provider(provider: str) -> Optional[str]:
    provider_name = provider.split("/", 1)[0].lower()
    if provider_name == "litellm_proxy":
        return os.getenv("LITELLM_BASE_URL")
    return os.getenv("LLM_BASE_URL")


def _raw_items_from_extracted_data(data: Any) -> List[dict]:
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict) and isinstance(data.get("item"), dict):
        raw_items = [data["item"]]
    elif isinstance(data, dict) and isinstance(data.get("items"), list):
        raw_items = data["items"]
    elif isinstance(data, dict):
        raw_items = [data]
    else:
        raw_items = []
    return [item for item in raw_items if isinstance(item, dict)]


def _first_present(*values: Any) -> Optional[Any]:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _fallback_item_for_link(source_link: Dict[str, Any], link_type: Optional[str]) -> dict:
    normalised_type = _normalise_link_type(link_type)
    title = _first_present(source_link.get("title"), source_link.get("url"), "Untitled")
    if normalised_type == "grant":
        return {
            "category": "grant",
            "title": title,
            "description": source_link.get("context_summary"),
            "funding_amount": None,
            "eligibility": None,
            "start_date": None,
            "end_date": None,
            "application_mode": None,
            "application_url": source_link.get("url"),
            "main_media_url": None,
        }
    return {
        "category": "news",
        "title": title,
        "summary": source_link.get("context_summary"),
        "published_date": source_link.get("publish_date"),
        "publisher": None,
        "url": source_link.get("url"),
        "main_media_url": None,
    }


def _normalise_grant_item(source_link: Dict[str, Any], item: Dict[str, Any]) -> dict:
    return {
        "category": "grant",
        "title": item.get("title"),
        "description": item.get("description"),
        "funding_amount": item.get("funding_amount"),
        "eligibility": item.get("eligibility"),
        "start_date": item.get("start_date"),
        "end_date": item.get("end_date"),
        "application_mode": item.get("application_mode"),
        "application_url": item.get("application_url"),
        "main_media_url": item.get("main_media_url"),
    }


def _merge_grant_items(source_link: Dict[str, Any], candidates: List[dict]) -> dict:
    fallback = _fallback_item_for_link(source_link, "grant")
    merged = {
        "category": "grant",
        "title": None,
        "description": None,
        "funding_amount": None,
        "eligibility": None,
        "start_date": None,
        "end_date": None,
        "application_mode": None,
        "application_url": None,
        "main_media_url": None,
    }
    for candidate in candidates:
        normalised = _normalise_grant_item(source_link, candidate)
        for key, value in normalised.items():
            if key == "category":
                continue
            if merged.get(key) in (None, "") and value not in (None, ""):
                merged[key] = value
    for key, value in fallback.items():
        if key == "category":
            continue
        if merged.get(key) in (None, ""):
            merged[key] = value
    return merged


def _normalise_news_item(source_link: Dict[str, Any], item: Dict[str, Any]) -> dict:
    return {
        "category": "news",
        "title": item.get("title"),
        "summary": _first_present(item.get("summary"), item.get("description")),
        "published_date": item.get("published_date"),
        "publisher": _first_present(item.get("publisher"), item.get("source")),
        "url": item.get("url"),
        "main_media_url": item.get("main_media_url"),
    }


def _merge_news_items(source_link: Dict[str, Any], candidates: List[dict]) -> dict:
    normalised_items = [_normalise_news_item(source_link, item) for item in candidates]
    if not normalised_items:
        return _fallback_item_for_link(source_link, "news")

    first = normalised_items[0]
    summaries: List[str] = []
    for item in normalised_items:
        summary = item.get("summary")
        if summary and summary not in summaries:
            summaries.append(summary)

    return {
        "category": "news",
        "title": _first_present(
            next((item.get("title") for item in normalised_items if item.get("title")), None),
            source_link.get("title"),
            source_link.get("url"),
        ),
        "summary": " ".join(summaries) if summaries else source_link.get("context_summary"),
        "published_date": _first_present(
            next((item.get("published_date") for item in normalised_items if item.get("published_date")), None),
            source_link.get("publish_date"),
        ),
        "publisher": next((item.get("publisher") for item in normalised_items if item.get("publisher")), None),
        "url": _first_present(
            next((item.get("url") for item in normalised_items if item.get("url")), None),
            source_link.get("url"),
        ),
        "main_media_url": next((item.get("main_media_url") for item in normalised_items if item.get("main_media_url")), None),
    }


def _single_item_for_link(source_link: Dict[str, Any], candidates: List[dict]) -> dict:
    normalised_type = _normalise_link_type(source_link.get("type"))
    if normalised_type == "grant":
        return _merge_grant_items(source_link, candidates)
    return _merge_news_items(source_link, candidates)


def _coerce_extracted_data(data: Any, link_type: Optional[str], source_url: str) -> Dict[str, List[dict]]:
    """Normalise provider output to exactly one item for the URL."""
    source_link = {"url": source_url, "type": link_type}
    normalised_type = _normalise_link_type(link_type)
    candidates = _raw_items_from_extracted_data(data)
    if normalised_type is None:
        return {"items": candidates}

    return {"items": [_single_item_for_link(source_link, candidates)]}


# ---------------------------------------------------------------------------
# 2. CORE EXTRACTION FUNCTION
# ---------------------------------------------------------------------------
async def extract_from_url(
        url: str,
        instruction: Optional[str] = None,
        link_type: Optional[str] = None,
        provider: str = "litellm_proxy/lite",
        input_format: str = "markdown",  # the input file type being passed to LLM
        chunk_token_threshold: int = 1000,
        overlap_rate: float = 0.1,  # 10% overlap so info isn't lost between chunks
        apply_chunking: bool = True,
        temperature: float = 0.0,  # controls creativity of LLM
        max_tokens: int = 5000,  # maximum number of tokens LLM can return in answer
        headless: bool = True,  # causes the browser to run invisibly in bg
) -> dict:
    """
    Crawl `url` and extract structured JSON data from it using an LLM.

    Parameters mirror the ones documented under "4. Key Parameters" in the
    Crawl4AI LLM Strategies docs.

    Returns
    -------
    dict with keys:
        "success": bool
        "data": parsed JSON (list/dict) if success else None
        "error": error message if not success
        "usage": token usage info (may be partial depending on provider)
    """

    if _CRAWL4AI_IMPORT_ERROR is not None:
        raise ImportError(
            "crawl4ai is required for extraction. Install project dependencies "
            "and run `crawl4ai-setup` before executing the extraction node."
        ) from _CRAWL4AI_IMPORT_ERROR

    normalised_type = _normalise_link_type(link_type)
    extraction_instruction = instruction or _instruction_for_type(normalised_type)

    # -- 2a. Configure the LLM provider (provider-agnostic via LiteLLM) -----
    # Format is "<provider>/<model_name>", e.g. "openai/gpt-4o-mini",
    # "ollama/llama2", "anthropic/claude-3-5-sonnet-latest", etc.
    api_token = _api_token_for_provider(provider)
    if not api_token:
        raise EnvironmentError(
            f"Missing API key for provider {provider!r}. Set LITELLM_API_KEY "
            "for litellm_proxy, GROQ_API_KEY for Groq, or OPENAI_API_KEY for OpenAI."
        )

    llm_config_kwargs = {
        "provider": provider,
        "api_token": api_token,
    }
    base_url = _base_url_for_provider(provider)
    if base_url:
        llm_config_kwargs["base_url"] = base_url

    llm_config = LLMConfig(**llm_config_kwargs)

    # -- 2b. Build the LLM extraction strategy -------------------------------
    llm_strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        schema=_schema_for_type(normalised_type),  # JSON schema from Pydantic model
        extraction_type="schema",  # "schema" -> strict JSON conforming to schema
        instruction=extraction_instruction,  # what to extract, in plain English
        chunk_token_threshold=chunk_token_threshold,  # max tokens per chunk sent to the LLM
        overlap_rate=overlap_rate,  # overlap between chunks to preserve context
        apply_chunking=apply_chunking,  # auto-split long pages into chunks
        input_format=input_format,  # "markdown" | "html" | "fit_markdown"
        extra_args={"temperature": temperature, "max_tokens": max_tokens},
        verbose=True,
    )

    # -- 2c. Strategy goes inside CrawlerRunConfig, NOT directly in arun() --
    markdown_generator = None
    if DefaultMarkdownGenerator is not None:
        try:
            markdown_generator = DefaultMarkdownGenerator(
                options={"ignore_images": False}
            )
        except Exception:
            markdown_generator = None

    crawl_config_kwargs = {
        "extraction_strategy": llm_strategy,
        "cache_mode": CacheMode.BYPASS,  # always fetch fresh content, don't use cache
        "exclude_external_images": False,
        "excluded_tags": ["script", "style", "header", "footer", "nav", "svg", "iframe"],
        "magic": True,  # Enables lazy loading, scrolling, and advanced bot evasion to ensure images load
    }
    if markdown_generator is not None:
        crawl_config_kwargs["markdown_generator"] = markdown_generator

    crawl_config = CrawlerRunConfig(**crawl_config_kwargs)

    # -- 2d. Browser configuration (headless crawler) ------------------------
    browser_cfg = BrowserConfig(headless=headless)

    # -- 2e. Run the crawl + extraction --------------------------------------
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)

        # Always show token usage so you can monitor cost/consumption.
        llm_strategy.show_usage()

        if not result.success:
            return {
                "success": False,
                "data": None,
                "error": result.error_message,
                "usage": getattr(llm_strategy, "total_usage", None),
            }

        # `result.extracted_content` is a JSON string produced by the LLM.
        # Parse it defensively: schema extraction is best-effort — the model
        # can occasionally return invalid or partial JSON (see docs, section
        # "10. Best Practices & Caveats").
        try:
            data = json.loads(result.extracted_content)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "data": None,
                "error": f"Failed to parse LLM output as JSON: {e}",
                "usage": getattr(llm_strategy, "total_usage", None),
            }

        data = _coerce_extracted_data(data, normalised_type, url)

        return {
            "success": True,
            "data": data,
            "error": None,
            "usage": getattr(llm_strategy, "total_usage", None),
        }


def _item_from_result(source_link: Dict[str, Any], extraction_result: dict) -> dict:
    data = extraction_result.get("data") or {}
    if isinstance(data, dict):
        extracted_items = data.get("items", [])
    else:
        extracted_items = []

    return _single_item_for_link(source_link, extracted_items)


async def extract_link(
        source_link: Dict[str, Any],
        provider: str = "litellm_proxy/lite",
        input_format: str = "markdown",
        chunk_token_threshold: int = 1200,
        overlap_rate: float = 0.1,
        apply_chunking: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        headless: bool = True,
) -> dict:
    """Extract one discovered link and return exactly one schema-shaped item."""
    url = source_link.get("url")
    link_type = _normalise_link_type(source_link.get("type"))
    if not url:
        return _fallback_item_for_link(source_link, link_type)

    result = await extract_from_url(
        url=url,
        link_type=link_type,
        provider=provider,
        input_format=input_format,
        chunk_token_threshold=chunk_token_threshold,
        overlap_rate=overlap_rate,
        apply_chunking=apply_chunking,
        temperature=temperature,
        max_tokens=max_tokens,
        headless=headless,
    )
    if not result.get("success"):
        return _fallback_item_for_link(source_link, link_type)
    return _item_from_result(source_link, result)


async def extract_many_from_links(
        source_links: Sequence[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = OUTPUT_FILE,
        provider: str = "litellm_proxy/lite",
        input_format: str = "markdown",
        chunk_token_threshold: int = 1200,
        overlap_rate: float = 0.1,
        apply_chunking: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        headless: bool = True,
) -> Dict[str, Any]:
    """
    Process all discovered URLs and optionally save one combined JSON file.

    The output contains one schema-shaped object per discovered URL:
    {"items": [{...}, {...}]}
    """
    items: List[dict] = []
    for source_link in source_links:
        items.append(
            await extract_link(
                source_link=source_link,
                provider=provider,
                input_format=input_format,
                chunk_token_threshold=chunk_token_threshold,
                overlap_rate=overlap_rate,
                apply_chunking=apply_chunking,
                temperature=temperature,
                max_tokens=max_tokens,
                headless=headless,
            )
        )

    output = {"items": items}

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    return output


def links_from_agent_output(payload: Any) -> List[Dict[str, Any]]:
    """Parse the ReAct node JSON shape and return normalised {url, type, ...} records."""
    if isinstance(payload, list):
        raw_links = payload
    elif isinstance(payload, dict):
        raw_links = (
            payload.get("valid_links")
            or payload.get("links")
            or payload.get("results")
            or []
        )
    else:
        raw_links = []

    links: List[Dict[str, Any]] = []
    seen_urls = set()
    for raw_link in raw_links:
        if not isinstance(raw_link, dict):
            continue
        url = raw_link.get("url")
        raw_type = raw_link.get("type") or raw_link.get("category")
        if not url or not raw_type:
            continue
        try:
            link_type = _normalise_link_type(raw_type)
        except ValueError:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        links.append({
            "title": raw_link.get("title"),
            "url": url,
            "type": link_type,
            "context_summary": raw_link.get("context_summary"),
            "publish_date": raw_link.get("publish_date"),
        })
    return links


# ---------------------------------------------------------------------------
# 3. COMMAND-LINE ENTRY POINT
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured JSON from a URL using Crawl4AI's LLMExtractionStrategy."
    )
    parser.add_argument("url", nargs="?", help="The URL to crawl and extract data from")
    parser.add_argument(
        "--type",
        dest="link_type",
        choices=["grant", "news"],
        default=None,
        help="The discovered type for this URL. Used to select the extraction schema and prompt.",
    )
    parser.add_argument(
        "--links-json",
        default=None,
        help="Path to a ReAct node JSON output file containing valid_links for batch extraction.",
    )
    parser.add_argument(
        "--instruction",
        default=(
            DEFAULT_EXTRACTION_INSTRUCTION
        ),
        help="Natural-language instruction telling the LLM what to extract",
    )
    parser.add_argument(
        "--provider",
        default="litellm_proxy/lite",
        help='LLM provider string, e.g. "litellm_proxy/lite"',
    )
    parser.add_argument(
        "--input-format",
        default="markdown",
        choices=["markdown", "html", "fit_markdown"],
        help="Which crawled content form is passed to the LLM",
    )
    parser.add_argument("--chunk-token-threshold", type=int, default=1200)
    parser.add_argument("--overlap-rate", type=float, default=0.1)
    parser.add_argument("--no-chunking", action="store_true", help="Disable automatic chunking")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Where to save the JSON result")
    return parser.parse_args()


async def main():
    args = parse_args()

    if args.links_json:
        with open(args.links_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        links = links_from_agent_output(payload)
        output = await extract_many_from_links(
            links,
            output_path=args.output,
            provider=args.provider,
            input_format=args.input_format,
            chunk_token_threshold=args.chunk_token_threshold,
            overlap_rate=args.overlap_rate,
            apply_chunking=not args.no_chunking,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(f"\nSaved batch extraction JSON to: {args.output}")
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if not args.url:
        raise SystemExit("Provide a URL, or pass --links-json for batch extraction.")

    output = await extract_many_from_links(
        [{"url": args.url, "type": args.link_type}],
        output_path=args.output,
        provider=args.provider,
        input_format=args.input_format,
        chunk_token_threshold=args.chunk_token_threshold,
        overlap_rate=args.overlap_rate,
        apply_chunking=not args.no_chunking,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(f"\nSaved extraction JSON to: {args.output}")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
