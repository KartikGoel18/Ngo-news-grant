"""Run the complete Impact Weaver discovery -> extraction -> rerank pipeline.

Flow:
1. Call the ReAct discovery node (`ReAct_Node/discovery_agent.py`).
2. Parse its JSON output and collect each discovered URL with its `type`.
3. Pass all `{url, type}` records into `Extraction_Node/extraction_node.py`.
4. Return the combined extraction output, optionally saving it to JSON.
5. Rerank the extracted items with `Rerank_Node/rerank_node.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parent
REACT_DIR = ROOT_DIR / "ReAct_Node"
EXTRACTION_DIR = ROOT_DIR / "Extraction_Node"
RERANK_DIR = ROOT_DIR / "Rerank_Node"
AGENT_MODULE_PATH = REACT_DIR / "discovery_agent.py"
EXTRACT_MODULE_PATH = EXTRACTION_DIR / "extraction_node.py"
RERANK_MODULE_PATH = RERANK_DIR / "rerank_node.py"
AGENT_OUTPUT_FILE = REACT_DIR / "discovered_links.json"
EXTRACT_OUTPUT_FILE = EXTRACTION_DIR / "extracted_items.json"
RERANK_OUTPUT_FILE = RERANK_DIR / "reranked_items.json"
RERANK_GRANTS_OUTPUT_FILE = RERANK_DIR / "scored_grant_items.json"
RERANK_NEWS_OUTPUT_FILE = RERANK_DIR / "scored_news_items.json"


def _load_module(module_name: str, module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(f"Module not found: {module_path}")

    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _model_or_dict_to_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected a Pydantic model or dict, got {type(value).__name__}.")


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _optional_path(value: Optional[str]) -> Optional[Path]:
    return Path(value) if value else None


def _items_only_payload(value: Any) -> Dict[str, Any]:
    payload = _model_or_dict_to_dict(value)
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Expected rerank output to contain an `items` list.")
    return {"items": items}


def _save_json_if_requested(path: Optional[Path], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_agent(master_query: Optional[str], output_path: Optional[Path]) -> Dict[str, Any]:
    agent_module = _load_module("impact_weaver_agents", AGENT_MODULE_PATH)

    if not hasattr(agent_module, "run_react_discovery"):
        raise AttributeError(f"{AGENT_MODULE_PATH} must expose run_react_discovery().")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {"output_path": output_path}
    if master_query:
        kwargs["master_query"] = master_query

    discovery_output = agent_module.run_react_discovery(**kwargs)
    if discovery_output is None:
        raise ValueError(
            "Discovery agent did not return valid structured JSON. "
            "Check the discovery agent logs for the raw model output and validation error."
        )
    payload = _model_or_dict_to_dict(discovery_output)

    _save_json_if_requested(output_path, payload)

    return payload


def load_extraction_module():
    return _load_module("impact_weaver_extraction_node", EXTRACT_MODULE_PATH)


def load_rerank_module():
    return _load_module("impact_weaver_rerank_node", RERANK_MODULE_PATH)


def parse_agent_links(agent_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    extraction_module = load_extraction_module()
    if not hasattr(extraction_module, "links_from_agent_output"):
        raise AttributeError(f"{EXTRACT_MODULE_PATH} must expose links_from_agent_output().")
    links = extraction_module.links_from_agent_output(agent_payload)
    if not links:
        raise ValueError("Agent output did not contain any valid links with both `url` and `type`.")
    return links


async def run_extraction(
        links: List[Dict[str, Any]],
        output_path: Optional[Path],
        provider: str,
        input_format: str,
        chunk_token_threshold: int,
        overlap_rate: float,
        apply_chunking: bool,
        temperature: float,
        max_tokens: int,
) -> Dict[str, Any]:
    extraction_module = load_extraction_module()
    if not hasattr(extraction_module, "extract_many_from_links"):
        raise AttributeError(f"{EXTRACT_MODULE_PATH} must expose extract_many_from_links().")

    return await extraction_module.extract_many_from_links(
        links,
        output_path=output_path,
        provider=provider,
        input_format=input_format,
        chunk_token_threshold=chunk_token_threshold,
        overlap_rate=overlap_rate,
        apply_chunking=apply_chunking,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _prepare_extracted_data_for_rerank(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt extraction output to the reranker schema.

    extraction_node.py emits `publisher` for news records, while rerank_node.py still
    names that field `source`. Keep the saved extraction file unchanged and add
    `source` only to the in-memory copy passed to reranking.
    """
    items = extracted_data.get("items")
    if not isinstance(items, list):
        raise ValueError("Extraction output must be a JSON object with an `items` list.")

    normalised_items: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        next_item = dict(item)
        category = next_item.get("category")

        if category == "grant":
            next_item["title"] = (
                next_item.get("title")
                or next_item.get("application_url")
                or next_item.get("url")
                or "Untitled grant"
            )
            next_item["description"] = next_item.get("description") or ""
            next_item["start_date"] = next_item.get("start_date") or ""
            normalised_items.append(next_item)
        elif category == "news":
            if not next_item.get("source"):
                next_item["source"] = next_item.get("publisher")
            next_item["title"] = next_item.get("title") or next_item.get("url") or "Untitled news"
            next_item["summary"] = next_item.get("summary") or next_item.get("content") or ""
            next_item["published_date"] = next_item.get("published_date") or ""
            next_item["source"] = next_item.get("source") or ""
            next_item["url"] = next_item.get("url") or ""
            normalised_items.append(next_item)

    return {"items": normalised_items}


def run_rerank_outputs(
        extracted_data: Dict[str, Any],
        output_path: Optional[Path],
        grants_output_path: Optional[Path],
        news_output_path: Optional[Path],
        query: str,
        top_fraction: float,
        top_k: Optional[int],
        per_category_top: bool,
        fusion_method: str,
        use_semantic_models: bool,
        model_cache_dir: str,
        allow_remote_model_downloads: bool,
) -> Dict[str, Dict[str, Any]]:
    rerank_module = load_rerank_module()
    required_attrs = ("RerankConfig", "RerankerNode")
    for attr in required_attrs:
        if not hasattr(rerank_module, attr):
            raise AttributeError(f"{RERANK_MODULE_PATH} must expose {attr}.")

    rerank_input = _prepare_extracted_data_for_rerank(extracted_data)
    config = rerank_module.RerankConfig(
        top_fraction=top_fraction,
        top_k=top_k,
        per_category_top=per_category_top,
        fusion_method=fusion_method,
        use_bi_encoder=use_semantic_models,
        use_cross_encoder=use_semantic_models,
        model_cache_dir=model_cache_dir,
        allow_remote_model_downloads=allow_remote_model_downloads,
    )
    all_grants, all_news, result = rerank_module.RerankerNode(config).rerank_with_all(rerank_input, query)
    grants_payload = _items_only_payload(all_grants)
    news_payload = _items_only_payload(all_news)
    selected_payload = _items_only_payload(result)

    _save_json_if_requested(output_path, selected_payload)
    _save_json_if_requested(grants_output_path, grants_payload)
    _save_json_if_requested(news_output_path, news_payload)

    return {
        "selected": selected_payload,
        "all_grants": grants_payload,
        "all_news": news_payload,
    }


def run_rerank(
        extracted_data: Dict[str, Any],
        output_path: Optional[Path],
        grants_output_path: Optional[Path],
        news_output_path: Optional[Path],
        query: str,
        top_fraction: float,
        top_k: Optional[int],
        per_category_top: bool,
        fusion_method: str,
        use_semantic_models: bool,
        model_cache_dir: str,
        allow_remote_model_downloads: bool,
) -> Dict[str, Any]:
    return run_rerank_outputs(
        extracted_data=extracted_data,
        output_path=output_path,
        grants_output_path=grants_output_path,
        news_output_path=news_output_path,
        query=query,
        top_fraction=top_fraction,
        top_k=top_k,
        per_category_top=per_category_top,
        fusion_method=fusion_method,
        use_semantic_models=use_semantic_models,
        model_cache_dir=model_cache_dir,
        allow_remote_model_downloads=allow_remote_model_downloads,
    )["selected"]


async def run_pipeline_outputs(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    agent_output_path = _optional_path(args.agent_output)
    extract_output_path = _optional_path(args.extract_output)
    rerank_output_path = _optional_path(args.rerank_output)
    rerank_grants_output_path = _optional_path(args.rerank_grants_output)
    rerank_news_output_path = _optional_path(args.rerank_news_output)

    if args.skip_extract:
        if extract_output_path is None:
            raise ValueError("--skip-extract requires --extract-output to point to a saved extraction JSON file.")
        extracted_data = _load_json(extract_output_path)
    else:
        if args.skip_agent:
            if agent_output_path is None:
                raise ValueError("--skip-agent requires --agent-output to point to a saved discovery JSON file.")
            agent_payload = _load_json(agent_output_path)
        else:
            agent_payload = run_agent(args.master_query, agent_output_path)

        links = parse_agent_links(agent_payload)

        if args.limit is not None:
            links = links[:args.limit]

        extracted_data = await run_extraction(
            links,
            output_path=extract_output_path,
            provider=args.provider,
            input_format=args.input_format,
            chunk_token_threshold=args.chunk_token_threshold,
            overlap_rate=args.overlap_rate,
            apply_chunking=not args.no_chunking,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    return run_rerank_outputs(
        extracted_data,
        output_path=rerank_output_path,
        grants_output_path=rerank_grants_output_path,
        news_output_path=rerank_news_output_path,
        query=args.query,
        top_fraction=args.top_fraction,
        top_k=args.top_k,
        per_category_top=args.per_category_top,
        fusion_method=args.fusion_method,
        use_semantic_models=not args.no_semantic_rerank,
        model_cache_dir=args.model_cache_dir,
        allow_remote_model_downloads=args.allow_remote_model_downloads,
    )


async def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    outputs = await run_pipeline_outputs(args)
    return outputs["selected"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate ReAct discovery, Crawl4AI extraction, and reranking."
    )
    parser.add_argument(
        "--master-query",
        default=None,
        help="Optional custom master query for the ReAct discovery node.",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Use the existing agent JSON file instead of running live discovery.",
    )
    parser.add_argument(
        "--skip-extract",
        dest="skip_extract",
        action="store_true",
        help="Use the saved extraction JSON and run only the rerank node.",
    )
    parser.add_argument(
        "--agent-output",
        default=str(AGENT_OUTPUT_FILE),
        help="Where the ReAct discovery JSON is read from or written to.",
    )
    parser.add_argument(
        "--extract-output",
        default=str(EXTRACT_OUTPUT_FILE),
        help="Where the combined extraction JSON should be saved.",
    )
    parser.add_argument(
        "--rerank-output",
        default=str(RERANK_OUTPUT_FILE),
        help="Where the reranked JSON should be saved.",
    )
    parser.add_argument(
        "--rerank-grants-output",
        default=str(RERANK_GRANTS_OUTPUT_FILE),
        help="Where JSON containing every scored grant should be saved.",
    )
    parser.add_argument(
        "--rerank-news-output",
        default=str(RERANK_NEWS_OUTPUT_FILE),
        help="Where JSON containing every scored news item should be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of discovered links to extract.",
    )
    parser.add_argument(
        "--provider",
        default="litellm_proxy/lite",
        help='LiteLLM provider string, e.g. "litellm_proxy/lite".',
    )
    parser.add_argument(
        "--input-format",
        default="markdown",
        choices=["markdown", "html", "fit_markdown"],
        help="Which crawled content form is passed to the LLM.",
    )
    parser.add_argument("--chunk-token-threshold", type=int, default=1200)
    parser.add_argument("--overlap-rate", type=float, default=0.1)
    parser.add_argument("--no-chunking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument(
        "--query",
        default="renewable energy grants and funding opportunities for startups",
        help="The query/interest to rank extracted items against.",
    )
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--per-category-top", action="store_true")
    parser.add_argument("--fusion-method", choices=["weighted_sum", "rrf"], default="weighted_sum")
    parser.add_argument(
        "--model-cache-dir",
        default=str(RERANK_DIR / "model_cache"),
        help="Directory containing predownloaded reranker models.",
    )
    parser.add_argument(
        "--allow-remote-model-downloads",
        action="store_true",
        help="Allow runtime Hugging Face downloads if local models are missing.",
    )
    parser.add_argument(
        "--no-semantic-rerank",
        action="store_true",
        help="Skip Hugging Face sentence-transformer bi/cross-encoder model loading.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output = asyncio.run(run_pipeline(args))
    selected_count = len(output.get("items", []))
    print(
        "Pipeline complete: "
        f"kept {selected_count} selected item(s), "
        f"saved grants to {Path(args.rerank_grants_output)}, news to {Path(args.rerank_news_output)}, "
        f"and final results to {Path(args.rerank_output)}"
    )


if __name__ == "__main__":
    main()
