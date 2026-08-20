"""MongoDB persistence for Impact Weaver pipeline outputs."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Annotated, Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except ImportError:  # pragma: no cover - handled at runtime with a clear message
    MongoClient = None
    PyMongoError = Exception


load_dotenv()

DEFAULT_SELECTED_COLLECTION = "ai_news_grants"
DEFAULT_NOT_SELECTED_NEWS_COLLECTION = "ai_not_selected_news"
DEFAULT_NOT_SELECTED_GRANT_COLLECTION = "ai_not_selected_grants"


class MongoStoreError(RuntimeError):
    """Raised when MongoDB persistence cannot be completed."""


class MongoScoreBreakdown(BaseModel):
    """Score fields saved with each MongoDB document."""

    model_config = ConfigDict(extra="allow")

    bm25: float
    bi_encoder: float
    cross_encoder: float
    recency: float
    completeness: float
    grant_deadline: Optional[float] = None
    grant_amount: Optional[float] = None
    grant_eligibility_match: Optional[float] = None
    grant_indian_ngo_relevance: Optional[float] = None
    grant_is_indian_ngo_relevant: Optional[bool] = None
    grant_deadline_passed: Optional[bool] = None
    source_credibility: Optional[float] = None
    news_indian_ngo_relevance: Optional[float] = None


class MongoItemDocumentBase(BaseModel):
    """Common fields required on every item saved to MongoDB."""

    model_config = ConfigDict(extra="allow")

    title: str
    rank: Optional[int] = None
    score_breakdown: MongoScoreBreakdown
    pipeline_run_id: str
    selection_status: Literal["selected", "not_selected_news", "not_selected_grant"]
    saved_at: datetime
    saved_at_iso: str
    main_media_url: Optional[str] = None


class MongoGrantDocument(MongoItemDocumentBase):
    category: Literal["grant"]
    description: Optional[str] = None
    funding_amount: Optional[str] = None
    eligibility: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    application_mode: Optional[str] = None
    application_url: Optional[str] = None
    final_score: bool


class MongoNewsDocument(MongoItemDocumentBase):
    category: Literal["news"]
    summary: Optional[str] = None
    published_date: Optional[str] = None
    source: Optional[str] = None
    publisher: Optional[str] = None
    url: Optional[str] = None
    final_score: float


MongoItemDocument = Annotated[
    Union[MongoGrantDocument, MongoNewsDocument],
    Field(discriminator="category"),
]
MONGO_ITEM_DOCUMENT_ADAPTER = TypeAdapter(MongoItemDocument)


def _required_mongodb_uri() -> str:
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise MongoStoreError(
            "Missing MongoDB connection string. Set MONGODB_URI in the .env file."
        )
    return uri


def validate_mongo_config() -> None:
    """Validate local Mongo settings without opening a network connection."""
    if MongoClient is None:
        raise MongoStoreError(
            "pymongo is not installed. Install dependencies with `pip install -r requirements.txt`."
        )
    _required_mongodb_uri()


def _collection_names() -> Dict[str, str]:
    return {
        "selected": os.getenv("MONGODB_SELECTED_COLLECTION", DEFAULT_SELECTED_COLLECTION),
        "not_selected_news": os.getenv(
            "MONGODB_NOT_SELECTED_NEWS_COLLECTION",
            DEFAULT_NOT_SELECTED_NEWS_COLLECTION,
        ),
        "not_selected_grants": os.getenv(
            "MONGODB_NOT_SELECTED_GRANT_COLLECTION",
            DEFAULT_NOT_SELECTED_GRANT_COLLECTION,
        ),
    }


def _items_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise MongoStoreError("Expected rerank payload to contain an `items` list.")
    return [item for item in items if isinstance(item, dict)]


def _item_key(item: Dict[str, Any]) -> Tuple[str, str, str]:
    category = str(item.get("category") or "")
    url = str(item.get("url") or item.get("application_url") or "")
    title = str(item.get("title") or "")
    return category, url, title


def split_selected_and_not_selected(
    selected_payload: Dict[str, Any],
    all_grants_payload: Dict[str, Any],
    all_news_payload: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Split rerank outputs into selected, not-selected grants, and not-selected news."""
    selected_items = _items_from_payload(selected_payload)
    selected_keys = {_item_key(item) for item in selected_items}

    all_grants = _items_from_payload(all_grants_payload)
    all_news = _items_from_payload(all_news_payload)

    return {
        "selected_items": selected_items,
        "not_selected_grant_items": [
            item for item in all_grants if _item_key(item) not in selected_keys
        ],
        "not_selected_news_items": [
            item for item in all_news if _item_key(item) not in selected_keys
        ],
    }


def _documents_for_items(
    items: Iterable[Dict[str, Any]],
    *,
    pipeline_run_id: str,
    saved_at: datetime,
    selection_status: str,
) -> List[Dict[str, Any]]:
    saved_at_iso = saved_at.isoformat()
    documents: List[Dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        document = dict(item)
        document["pipeline_run_id"] = pipeline_run_id
        document["selection_status"] = selection_status
        document["saved_at"] = saved_at
        document["saved_at_iso"] = saved_at_iso
        try:
            validated_document = MONGO_ITEM_DOCUMENT_ADAPTER.validate_python(document)
        except ValidationError as exc:
            title = item.get("title") or item.get("url") or item.get("application_url") or "<untitled>"
            raise MongoStoreError(
                f"MongoDB document schema validation failed for {selection_status} "
                f"item #{index} ({title}): {exc}"
            ) from exc
        documents.append(validated_document.model_dump())
    return documents


def _insert_many_if_present(collection, documents: List[Dict[str, Any]]) -> int:
    if not documents:
        return 0
    result = collection.insert_many(documents)
    return len(result.inserted_ids)


def save_pipeline_outputs_to_mongo(
    *,
    selected_payload: Dict[str, Any],
    all_grants_payload: Dict[str, Any],
    all_news_payload: Dict[str, Any],
    pipeline_run_id: str,
) -> Dict[str, Any]:
    """Persist selected and not-selected rerank items to MongoDB collections."""
    validate_mongo_config()
    split_items = split_selected_and_not_selected(
        selected_payload,
        all_grants_payload,
        all_news_payload,
    )

    IST = timezone(timedelta(hours=5, minutes=30))
    saved_at = datetime.now(IST)
    collection_names = _collection_names()
    client = None
    try:
        client = MongoClient(_required_mongodb_uri(), serverSelectionTimeoutMS=10000)
        database = client.get_default_database()
        database_name = database.name

        selected_count = _insert_many_if_present(
            database[collection_names["selected"]],
            _documents_for_items(
                split_items["selected_items"],
                pipeline_run_id=pipeline_run_id,
                saved_at=saved_at,
                selection_status="selected",
            ),
        )
        not_selected_news_count = _insert_many_if_present(
            database[collection_names["not_selected_news"]],
            _documents_for_items(
                split_items["not_selected_news_items"],
                pipeline_run_id=pipeline_run_id,
                saved_at=saved_at,
                selection_status="not_selected_news",
            ),
        )
        not_selected_grants_count = _insert_many_if_present(
            database[collection_names["not_selected_grants"]],
            _documents_for_items(
                split_items["not_selected_grant_items"],
                pipeline_run_id=pipeline_run_id,
                saved_at=saved_at,
                selection_status="not_selected_grant",
            ),
        )
    except PyMongoError as exc:
        raise MongoStoreError(f"MongoDB save failed: {exc}") from exc
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    return {
        "pipeline_run_id": pipeline_run_id,
        "database": database_name,
        "saved_at_iso": saved_at.isoformat(),
        "collections": {
            collection_names["selected"]: selected_count,
            collection_names["not_selected_news"]: not_selected_news_count,
            collection_names["not_selected_grants"]: not_selected_grants_count,
        },
    }


def save_pipeline_outputs_to_mongo_with_logging(
    *,
    selected_payload: Dict[str, Any],
    all_grants_payload: Dict[str, Any],
    all_news_payload: Dict[str, Any],
    pipeline_run_id: str,
) -> None:
    """Background-task wrapper that logs Mongo errors instead of hiding them silently."""
    try:
        summary = save_pipeline_outputs_to_mongo(
            selected_payload=selected_payload,
            all_grants_payload=all_grants_payload,
            all_news_payload=all_news_payload,
            pipeline_run_id=pipeline_run_id,
        )
        print(f"MongoDB save complete: {summary}", file=sys.stderr)
    except Exception as exc:
        print(f"MongoDB save failed for run {pipeline_run_id}: {exc}", file=sys.stderr)
