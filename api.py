"""FastAPI entrypoint for running the Impact Weaver pipeline."""

from __future__ import annotations

import asyncio
import sys
import uuid
from typing import Any, Dict, Literal, Optional
import os
import secrets
import time

# pyrefly: ignore [missing-import]
from fastapi import BackgroundTasks, FastAPI, HTTPException, Depends, status
# pyrefly: ignore [missing-import]
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, field_validator

import pipeline
from mongo_store import (
    MongoStoreError,
    save_pipeline_outputs_to_mongo,
    save_pipeline_outputs_to_mongo_with_logging,
    split_selected_and_not_selected,
    validate_mongo_config,
)


security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    # Note: Added "" fallback to prevent TypeError if the .env variable is completely missing
    correct_username = secrets.compare_digest(credentials.username, os.getenv("API_USERNAME", ""))
    correct_password = secrets.compare_digest(credentials.password, os.getenv("API_PASSWORD", ""))
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

app = FastAPI(title="Impact Weaver Pipeline API", version="1.0.0", dependencies=[Depends(verify_credentials)])
PIPELINE_JOBS: Dict[str, Dict[str, Any]] = {}


class PipelineRunRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "master_query": None,
                "query": None,
                "limit": 10,
                "skip_agent": False,
                "skip_extract": False,
                "provider": "litellm_proxy/lite",
                "input_format": "markdown",
                "chunk_token_threshold": 1200,
                "overlap_rate": 0.1,
                "no_chunking": False,
                "temperature": 0.0,
                "max_tokens": 10000,
                "top_fraction": 0.1,
                "top_k": None,
                "per_category_top": False,
                "fusion_method": "weighted_sum",
                "no_semantic_rerank": False,
                "allow_remote_model_downloads": False,
                "save_to_mongo": True,
                "include_not_selected": False,
                "include_metadata": False,
            }
        }
    }

    master_query: Optional[str] = Field(
        None,
        description="Optional custom query for the discovery agent.",
    )
    query: Optional[str] = Field(
        None,
        description="Optional reranking query. Uses pipeline.py's default when omitted.",
    )
    limit: Optional[int] = Field(
        None,
        ge=1,
        description="Optional maximum number of discovered links to extract.",
    )
    skip_agent: bool = Field(
        False,
        description="Use the saved discovery JSON instead of running live discovery.",
    )
    skip_extract: bool = Field(
        False,
        description="Use the saved extraction JSON and run only reranking.",
    )
    provider: str = Field(
        "litellm_proxy/lite",
        description="LiteLLM provider string used by the extraction node.",
    )
    input_format: Literal["markdown", "html", "fit_markdown"] = "markdown"
    chunk_token_threshold: int = Field(1200, ge=1)
    overlap_rate: float = Field(0.1, ge=0.0, le=1.0)
    no_chunking: bool = False
    temperature: float = Field(0.0, ge=0.0)
    max_tokens: int = Field(1500, ge=1)
    top_fraction: float = Field(0.10, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1)
    per_category_top: bool = False
    fusion_method: Literal["weighted_sum", "rrf"] = "weighted_sum"
    no_semantic_rerank: bool = False
    allow_remote_model_downloads: bool = False
    save_to_mongo: bool = Field(
        True,
        description="Schedule MongoDB persistence as a FastAPI background task.",
    )
    include_not_selected: bool = Field(
        False,
        description="Include not-selected grant/news JSON in the API response.",
    )
    include_metadata: bool = Field(
        False,
        description="Wrap the selected JSON with run metadata.",
    )

    @field_validator("master_query", "query", mode="before")
    @classmethod
    def _normalise_optional_text(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        value = value.strip()
        if not value or value == "string":
            return None
        return value


def _pipeline_args_from_request(request: PipelineRunRequest, pipeline_run_id: str):
    args = pipeline.parse_args([])
    args.master_query = request.master_query
    if request.query:
        args.query = request.query
    args.limit = request.limit
    args.skip_agent = request.skip_agent
    args.skip_extract = request.skip_extract
    args.provider = request.provider
    args.input_format = request.input_format
    args.chunk_token_threshold = request.chunk_token_threshold
    args.overlap_rate = request.overlap_rate
    args.no_chunking = request.no_chunking
    args.temperature = request.temperature
    args.max_tokens = request.max_tokens
    args.top_fraction = request.top_fraction
    args.top_k = request.top_k
    args.per_category_top = request.per_category_top
    args.fusion_method = request.fusion_method
    args.no_semantic_rerank = request.no_semantic_rerank
    args.allow_remote_model_downloads = request.allow_remote_model_downloads

    if not request.skip_agent:
        args.agent_output = None
    if not request.skip_extract:
        args.extract_output = None
    args.rerank_output = None
    args.rerank_grants_output = None
    args.rerank_news_output = None
    return args


async def _run_pipeline_and_collect(
    request: PipelineRunRequest,
    pipeline_run_id: str,
) -> Dict[str, Any]:
    if request.save_to_mongo:
        validate_mongo_config()

    args = _pipeline_args_from_request(request, pipeline_run_id)
    
    start_time = time.perf_counter()
    pipeline_outputs = await pipeline.run_pipeline_outputs(args)
    end_time = time.perf_counter()
    
    total_runtime_seconds = round(end_time - start_time, 2)
    
    selected_payload = pipeline_outputs["selected"]
    all_grants_payload = pipeline_outputs["all_grants"]
    all_news_payload = pipeline_outputs["all_news"]
    split_items = split_selected_and_not_selected(
        selected_payload,
        all_grants_payload,
        all_news_payload,
    )
    return {
        "pipeline_run_id": pipeline_run_id,
        "total_runtime_seconds": total_runtime_seconds,
        "selected_payload": selected_payload,
        "all_grants_payload": all_grants_payload,
        "all_news_payload": all_news_payload,
        "split_items": split_items,
    }


def _response_for_request(
    request: PipelineRunRequest,
    run_data: Dict[str, Any],
    mongo_save: str,
) -> Dict[str, Any]:
    selected_payload = run_data["selected_payload"]
    if not request.include_metadata and not request.include_not_selected:
        return selected_payload

    response: Dict[str, Any] = {
        "pipeline_run_id": run_data["pipeline_run_id"],
        "total_runtime_seconds": run_data.get("total_runtime_seconds"),
        "selected": selected_payload,
        "mongo_save": mongo_save,
    }
    if request.include_not_selected:
        split_items = run_data["split_items"]
        response["not_selected"] = {
            "grant_items": split_items["not_selected_grant_items"],
            "news_items": split_items["not_selected_news_items"],
        }
    return response


def _error_detail(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return f"{type(exc).__name__}: {exc}"


def _run_async_for_pipeline(coro):
    """Run Crawl4AI/Playwright work in a Windows-compatible event loop."""
    if sys.platform == "win32" and hasattr(asyncio, "ProactorEventLoop"):
        loop = asyncio.ProactorEventLoop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()
    return asyncio.run(coro)


def _run_pipeline_background(request_data: Dict[str, Any], pipeline_run_id: str) -> None:
    print(f"Background pipeline job {pipeline_run_id} started.", file=sys.stderr)
    PIPELINE_JOBS[pipeline_run_id] = {"status": "running", "pipeline_run_id": pipeline_run_id}
    try:
        request = PipelineRunRequest(**request_data)
        run_data = _run_async_for_pipeline(_run_pipeline_and_collect(request, pipeline_run_id))
        mongo_summary = None
        if request.save_to_mongo:
            mongo_summary = save_pipeline_outputs_to_mongo(
                selected_payload=run_data["selected_payload"],
                all_grants_payload=run_data["all_grants_payload"],
                all_news_payload=run_data["all_news_payload"],
                pipeline_run_id=pipeline_run_id,
            )
            print(f"MongoDB save complete: {mongo_summary}", file=sys.stderr)

        PIPELINE_JOBS[pipeline_run_id] = {
            "status": "completed",
            "pipeline_run_id": pipeline_run_id,
            "total_runtime_seconds": run_data.get("total_runtime_seconds"),
            "selected": run_data["selected_payload"],
            "not_selected": {
                "grant_items": run_data["split_items"]["not_selected_grant_items"],
                "news_items": run_data["split_items"]["not_selected_news_items"],
            },
            "mongo_save": mongo_summary or "skipped",
        }
        print(f"Background pipeline job {pipeline_run_id} completed successfully.", file=sys.stderr)
    except Exception as exc:
        PIPELINE_JOBS[pipeline_run_id] = {
            "status": "failed",
            "pipeline_run_id": pipeline_run_id,
            "error": _error_detail(exc),
        }
        print(f"Background pipeline job {pipeline_run_id} failed: {exc}", file=sys.stderr)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/pipeline/run")
def run_pipeline_endpoint(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Run the full pipeline and return the selected-items JSON by default."""
    pipeline_run_id = uuid.uuid4().hex

    try:
        run_data = _run_async_for_pipeline(_run_pipeline_and_collect(request, pipeline_run_id))

        if request.save_to_mongo:
            background_tasks.add_task(
                save_pipeline_outputs_to_mongo_with_logging,
                selected_payload=run_data["selected_payload"],
                all_grants_payload=run_data["all_grants_payload"],
                all_news_payload=run_data["all_news_payload"],
                pipeline_run_id=pipeline_run_id,
            )

        return _response_for_request(
            request,
            run_data,
            mongo_save="scheduled" if request.save_to_mongo else "skipped",
        )

    except MongoStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Required file was not found: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=f"Configuration error: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline run failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/pipeline/run-background")
def run_pipeline_background_endpoint(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Schedule the full pipeline as a true FastAPI background task."""
    pipeline_run_id = uuid.uuid4().hex
    try:
        if request.save_to_mongo:
            validate_mongo_config()
    except MongoStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    PIPELINE_JOBS[pipeline_run_id] = {"status": "queued", "pipeline_run_id": pipeline_run_id}
    background_tasks.add_task(
        _run_pipeline_background,
        request.model_dump(),
        pipeline_run_id,
    )
    return {
        "status": "queued",
        "pipeline_run_id": pipeline_run_id,
        "status_url": f"/pipeline/jobs/{pipeline_run_id}",
    }


@app.get("/pipeline/jobs/{pipeline_run_id}")
def get_pipeline_job(pipeline_run_id: str) -> Dict[str, Any]:
    job = PIPELINE_JOBS.get(pipeline_run_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"No pipeline job found for pipeline_run_id={pipeline_run_id}",
        )
    return job
