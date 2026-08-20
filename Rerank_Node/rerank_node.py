"""
rerank_node.py
==============
A hybrid reranking node for the item list produced by the extraction node's
discriminated-union `LLMExtractionStrategy` pipeline (see GrantItem /
NewsItem / ExtractedItem / ExtractionResult there).

This is meant to be dropped in immediately AFTER extraction in a pipeline:

    extraction_node.py --url--> ExtractionResult (JSON) --query--> rerank_node.py --> RerankedResult (JSON)

WHY THIS VERSION IS DIFFERENT FROM A GENERIC RERANKER
----------------------------------------------------------------------------
The extraction node already stopped pretending grants and news share one shape
(it split `ExtractedItem` into `GrantItem` / `NewsItem` via a discriminated
union). This reranker takes that one step further: *ranking* a grant and a
ranking a news item are different judgment tasks, not the same scoring
function applied to two field sets. So every signal below is either:

  (a) SHARED but computed with category-aware inputs (recency reads
      `start_date` for a grant and `published_date` for a news item, using a
      different decay rate for each — see "RECENCY IS CATEGORY-AWARE"), or
  (b) CATEGORY-EXCLUSIVE (only meaningful for one category, `None` for the
      other, and only present in that category's weight profile).

SIGNAL INVENTORY
----------------------------------------------------------------------------
Shared (computed for every item, both categories):
  1. BM25 (lexical)      - cheap, exact-keyword aware; good for scheme
                            names, acronyms, place names a dense model may
                            blur. Weak on paraphrase/synonymy.
  2. Bi-encoder (dense)   - independent query/item embeddings, cosine sim.
                            Fast, catches semantic/paraphrase relevance BM25
                            misses; weaker on fine-grained relevance since
                            query and doc never attend to each other.
  3. Cross-encoder        - joint (query, item) forward pass through one
                            transformer; materially more accurate than the
                            bi-encoder, too slow for a big corpus but fine
                            over one page's worth of candidates (the
                            standard "retrieve cheap, rerank precise"
                            cascade).
  4. Recency              - freshness, but *not* one global decay curve —
                            see below.
  5. Completeness         - fraction of a *category-specific* optional
                            field set the extractor filled in. This is a
                            data-quality proxy, not a relevance signal: a
                            grant missing eligibility/deadline is a weaker
                            record than a fully-specified one, and a news
                            item missing source/date is a weaker record too
                            — but they're penalised against their own
                            field set, not a shared generic one.

Grant-exclusive:
  6. grant_deadline       - actionability of the application window, i.e.
                            is there still realistically time to apply.
  7. grant_amount         - size of the funding on offer, log-scaled and
                            corpus-relative.
  8. grant_eligibility_match - lexical overlap between the query and the
                            grant's `eligibility` text. Eligibility text is
                            usually terse hard-constraint language ("MSMEs
                            only", "women entrepreneurs", "Assam-based")
                            that a dense encoder tends to under-weight next
                            to a longer description — this is a cheap,
                            auditable complement, not a replacement, for
                            the semantic signals above.

News-exclusive:
  9. source_credibility   - checks the item's `source` (and, failing that,
                            the domain in `url`) against a configurable
                            allowlist of known outlets/government domains.
                            A proxy for "is this coverage or an
                            unverified blog/press-release mirror" — data
                            quality for news, the way completeness is data
                            quality for extraction thoroughness.

RECENCY IS CATEGORY-AWARE
----------------------------------------------------------------------------
The extraction node gives news items a real `published_date` field and grants a
real `start_date` (applications-open date) field — no more overloading one
date field across categories the way the old flat schema did. This reranker
uses:
    grants -> start_date, decayed with `grant_recency_half_life_days` (default
              60 — an open funding scheme doesn't go stale for weeks/months)
    news   -> published_date, decayed with `news_recency_half_life_days`
              (default 5 — news is relevant for days, not weeks)
Different half-lives are the whole point: applying a news-speed decay curve
to a grant would tank perfectly-actionable schemes just because they opened
three weeks ago, and applying a grant-speed decay to news would let month-old
stories rank alongside today's.

CATEGORY-SPECIFIC WEIGHT PROFILES
----------------------------------------------------------------------------
`RerankConfig` carries two separate weight dicts — `grant_weights` and
`news_weights` — rather than one shared dict with a few extra keys. This
lets you tune e.g. "recency matters a lot more for news than for grants"
(default news recency weight is higher) or "grant amount/deadline matter a
lot for grants but have no news analogue at all" without one category's
irrelevant knobs cluttering the other's config.

FUSION
----------------------------------------------------------------------------
Both `weighted_sum` and `rrf` (Reciprocal Rank Fusion) are supported, exactly
as in the previous version — see `RerankConfig.fusion_method`. Every item
picks its own weight profile (grant vs. news) at fusion time; missing
category-exclusive signals (e.g. a grant with no parseable eligibility text)
are excluded from that item's weighted sum and the remaining weights are
renormalised, so a data gap isn't punished as if it were low relevance.

REQUIREMENTS
----------------------------------------------------------------------------
    pip install rank_bm25 sentence-transformers python-dateutil pydantic

Each dependency is imported lazily / defensively so importing this module
for its Pydantic schemas alone never requires a GPU or a multi-GB
`sentence-transformers` install.

USAGE
----------------------------------------------------------------------------
    # As a library:
    from rerank_node import RerankerNode, RerankConfig
    node = RerankerNode(RerankConfig(top_fraction=0.10))
    result = node.rerank(extraction_result_dict, query="renewable energy grants for startups")

    # As a pipeline "node" (e.g. inside a LangGraph-style state dict):
    from rerank_node import rerank_node
    state = rerank_node(state, query=state["query"])

    # From the command line:
    python rerank_node.py extracted_items.json --query "renewable energy grants" --top-fraction 0.1
    python rerank_node.py --demo   # smoke-test on a small built-in synthetic dataset
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field

RERANK_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CACHE_DIR = RERANK_DIR / "model_cache"
BI_ENCODER_CACHE_NAME = "bi_encoder"
CROSS_ENCODER_CACHE_NAME = "cross_encoder"

# ---------------------------------------------------------------------------
# 0. OPTIONAL / HEAVY DEPENDENCIES — imported defensively.
#    rank_bm25              -> lexical BM25 scoring
#    python-dateutil        -> flexible date parsing
#    sentence-transformers  -> bi-encoder + cross-encoder (imported lazily,
#                               inside RerankerNode, since it drags in torch)
# ---------------------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

try:
    from dateutil import parser as _dateutil_parser
    _HAS_DATEUTIL = True
except ImportError:
    _HAS_DATEUTIL = False


# ---------------------------------------------------------------------------
# 1. INPUT SCHEMA - a local copy of extraction_node.py's GrantItem / NewsItem
#    / ExtractedItem / ExtractionResult, so this file can be dropped anywhere
#    and run standalone.
#
#    NOTE: a few fields below (e.g. GrantItem.description, GrantItem.start_date,
#    NewsItem.summary/published_date/source/url) are typed `str` with a
#    `None` default in the source file — a pre-existing quirk of
#    extraction_node.py, not something introduced here. Pydantic v2 doesn't
#    validate defaults unless asked to, so this doesn't error, but it does
#    mean these fields can genuinely come back `None` at runtime despite the
#    `str` annotation. Every helper below treats them defensively (falsy
#    check) rather than trusting the annotation.
# ---------------------------------------------------------------------------
class GrantItem(BaseModel):
    """Schema used when an extracted item is a grant / funding scheme."""
    category: Literal["grant"] = Field("grant", description="Fixed discriminator value for grant items")
    title: str = Field(..., description="The name of the grant or funding scheme")
    description: str = Field(None, description="A short summary of what the grant funds or its purpose")
    funding_amount: Optional[str] = Field(None, description="The grant amount, funding size, or value range, if stated")
    eligibility: Optional[str] = Field(None, description="Who is eligible to apply for this grant, if stated")
    start_date: str = Field(None, description="The date applications open, if present")
    end_date: Optional[str] = Field(None, description="The application deadline / closing date for the grant, if present")
    application_mode: Optional[str] = Field(None, description="Mode of application, such as online, offline, email, postal, or not specified")
    application_url: Optional[str] = Field(None, description="The URL for applying, but only if the application mode is online and a link is present")
    main_media_url: Optional[str] = Field(None, description="Link for the main media file (image or video) if it exists")


class NewsItem(BaseModel):
    """Schema used when an extracted item is a news item / announcement."""
    category: Literal["news"] = Field("news", description="Fixed discriminator value for news items")
    title: str = Field(..., description="The headline of the news item")
    summary: str = Field(None, description="A short summary of the news content")
    published_date: str = Field(None, description="The date the news item was published or posted, if present")
    source: str = Field(None, description="The publisher, department, or author of the news item, if stated")
    url: str = Field(None, description="A link to the full news article, if present")
    main_media_url: Optional[str] = Field(None, description="Link for the main media file (image or video) if it exists")


ExtractedItem = Union[GrantItem, NewsItem],Field(discriminator="category")


class ExtractionResult(BaseModel):
    """Top-level container matching extraction_node.py's output: `{"items": [...]}`."""
    items: List[Union[GrantItem, NewsItem]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. OUTPUT SCHEMA — GrantItem/NewsItem plus rank + scores. `RerankedItem`
#    can't simply subclass a Union alias the way the old single-model
#    version could, so each category gets its own reranked subclass, joined
#    back into a discriminated union for the output container.
# ---------------------------------------------------------------------------
class ScoreBreakdown(BaseModel):
    """Every individual signal that fed into an item's final_score, kept on
    the item itself so the ranking is auditable/debuggable rather than a
    black box. Category-exclusive fields are `None` on items where they
    don't apply, rather than omitted, so the shape is predictable to
    consume downstream."""
    bm25: float
    bi_encoder: float
    cross_encoder: float
    recency: float
    completeness: float
    # grant-only
    grant_deadline: Optional[float] = None
    grant_amount: Optional[float] = None
    grant_eligibility_match: Optional[float] = None
    grant_indian_ngo_relevance: Optional[float] = None
    grant_is_indian_ngo_relevant: Optional[bool] = None
    grant_deadline_passed: Optional[bool] = None
    # news-only
    source_credibility: Optional[float] = None
    news_indian_ngo_relevance: Optional[float] = None


class RerankedGrantItem(GrantItem):
    rank: Optional[int] = Field(None, description="Unused for grants because grants are accepted/rejected, not ranked.")
    final_score: bool = Field(..., description="True only when the grant is relevant to Indian NGOs and its known deadline has not passed.")
    score_breakdown: ScoreBreakdown


class RerankedNewsItem(NewsItem):
    rank: int = Field(..., description="1-indexed rank after reranking (1 = most relevant)")
    final_score: float = Field(..., description="Final fused relevance score in [0, 1] used to produce `rank`")
    score_breakdown: ScoreBreakdown


RerankedItem = Union[RerankedGrantItem, RerankedNewsItem]


class RerankedResult(BaseModel):
    """Top-level output container."""
    query: str
    items: List[Union[RerankedGrantItem, RerankedNewsItem]]
    total_candidates: int
    returned_count: int
    top_fraction: float
    fusion_method: str
    grant_weights: Dict[str, float]
    news_weights: Dict[str, float]


# ---------------------------------------------------------------------------
# 3. HYPERPARAMETERS — every tunable knob lives here, nowhere else.
# ---------------------------------------------------------------------------
class RerankConfig(BaseModel):
    # --- how many results to keep ---
    top_fraction: float = Field(0.50, description="Fraction of candidates to keep, e.g. 0.10 = top 10%")
    top_k: Optional[int] = Field(3, description="If set, keep exactly this many items instead of a fraction")
    min_items_returned: int = Field(1, description="Never return fewer than this many items per selection group, even for tiny candidate sets")
    per_category_top: bool = Field(
        False,
        description=(
            "If True, apply top_fraction/top_k independently to grants and to news, then merge — "
            "so one category can't crowd the other out of the results just because its signals "
            "score higher on average. If False (default), one shared cutoff is applied across the "
            "whole pooled, cross-category ranking."
        ),
    )

    # --- fusion strategy ---
    fusion_method: Literal["weighted_sum", "rrf"] = Field(
        "weighted_sum",
        description=(
            "'weighted_sum' gives direct, interpretable control per signal (good default here since "
            "grant deadline/amount/eligibility and news source-credibility each need their own "
            "explicit weight). 'rrf' (Reciprocal Rank Fusion) is scale-free and more robust when "
            "component score distributions are wildly different, at the cost of losing magnitude info."
        ),
    )
    rrf_k: int = Field(60, description="RRF smoothing constant; 60 is the standard value from the original RRF paper's recommendation range")

    # --- category-specific weight profiles ---
    grant_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "bm25": 0.15,
            "bi_encoder": 0.15,
            "cross_encoder": 0.15,
            "recency": 0.10,
            "completeness": 0.10,
            "grant_deadline": 0.15,
            "grant_amount": 0.10,
            "grant_eligibility_match": 0.10,
        },
        description="Signal weights applied to GrantItem rows. Non-grant keys (source_credibility) are ignored here.",
    )
    news_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "bm25": 0.20,
            "bi_encoder": 0.20,
            "cross_encoder": 0.25,
            "recency": 0.03,
            "completeness": 0.20,
            "source_credibility": 0.02,
            "news_indian_ngo_relevance": 0.10,
        },
        description=(
            "Signal weights applied to NewsItem rows. Relevance (BM25, bi-encoder and cross-encoder) "
            "and extraction completeness dominate; recency is intentionally a minor signal."
        ),
    )

    # --- models ---
    use_bi_encoder: bool = Field(True, description="If False, skip sentence-transformer bi-encoder scoring and use neutral scores.")
    use_cross_encoder: bool = Field(True, description="If False, skip sentence-transformer cross-encoder scoring and use neutral scores.")
    model_cache_dir: str = Field(
        default_factory=lambda: str(DEFAULT_MODEL_CACHE_DIR),
        description="Project-local directory containing predownloaded sentence-transformer models.",
    )
    allow_remote_model_downloads: bool = Field(
        False,
        description=(
            "If True, sentence-transformers may contact Hugging Face when local models are missing. "
            "Default is False so runtime reranking uses only predownloaded local models."
        ),
    )
    bi_encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = Field(
        "cpu",
        description=(
            "Device sentence-transformers loads the bi-encoder/cross-encoder onto. "
            "'cpu' is the safe, dependency-free default. Set to 'cuda' if a GPU + CUDA-enabled "
            "torch build is available (materially faster for larger candidate batches), or 'mps' "
            "on Apple Silicon. Not validated here — an invalid value will surface as an error from "
            "torch/sentence-transformers itself when the model is loaded."
        ),
    )

    # --- cascade optimisation ---
    cross_encoder_prefilter_fraction: float = Field(
        1.0,
        description=(
            "Only run the (expensive) cross-encoder on the top X fraction of items ranked by the "
            "cheap BM25+bi-encoder combo; the rest get a cross-encoder score of 0. 1.0 (default) "
            "scores every candidate. Lower this (e.g. 0.3) if candidate sets get into the "
            "hundreds/thousands and cross-encoder latency matters."
        ),
    )

    # --- recency (category-aware; see module docstring) ---
    grant_recency_half_life_days: float = Field(60.0, description="Days for a grant's recency score (based on start_date) to decay to 0.5")
    news_recency_half_life_days: float = Field(5.0, description="Days for a news item's recency score (based on published_date) to decay to 0.5")
    recency_default_score: float = Field(0.5, description="Neutral score used when no date could be parsed, so missing data isn't punished")

    # --- grants ---
    grant_indian_ngo_relevance_threshold: float = Field(
        0.45,
        description="Minimum Indian-NGO relevance score required for a grant's boolean final_score to be true.",
    )
    grant_min_days_needed: float = Field(3.0, description="Below this many days-remaining, a deadline is treated as unactionable (score 0)")
    grant_ideal_days_remaining: float = Field(21.0, description="Days-remaining that scores highest — enough runway to prepare a strong application")
    grant_days_std: float = Field(20.0, description="Spread of the deadline 'sweet spot' bump around grant_ideal_days_remaining")
    higher_grant_amount_is_better: bool = Field(True, description="If True, larger grants rank higher; flip to False to surface smaller/easier-to-win grants first")

    # --- news ---
    reputable_sources: List[str] = Field(
        default_factory=lambda: [
            "Reuters - reuters.com",
            "Associated Press (AP) - apnews.com",
            "Agence France-Presse (AFP) - afp.com",
            "BBC - bbc.com",
            "BBC - bbc.co.uk",
            "The Guardian - theguardian.com",
            "The New York Times - nytimes.com",
            "The Washington Post - washingtonpost.com",
            "NPR - npr.org",
            "Al Jazeera English - aljazeera.com",
            "Deutsche Welle (DW) - dw.com",
            "France 24 - france24.com",
            "CBC News (Canada) - cbc.ca",
            "ABC News (Australia) - abc.net.au",
            "South China Morning Post - scmp.com",
            "Le Monde (France) - lemonde.fr",
            "El País (Spain) - elpais.com",
            "Bloomberg - bloomberg.com",
            "Financial Times - ft.com",
            "The Wall Street Journal - wsj.com",
            "The Economist - economist.com",
            "Press Information Bureau (PIB) - pib.gov.in",
            "National Portal of India - india.gov.in",
            "Reserve Bank of India - rbi.org.in",
            "Ministry of Statistics (MoSPI) - mospi.gov.in",
            "Press Trust of India (PTI) - ptinews.com",
            "Asian News International (ANI) - aninews.in",
            "Indo-Asian News Service (IANS) - ians.in",
            "The Hindu - thehindu.com",
            "Hindustan Times - hindustantimes.com",
            "Times of India - timesofindia.indiatimes.com",
            "Indian Express - indianexpress.com",
            "NDTV - ndtv.com",
            "India Today - indiatoday.in",
            "Deccan Herald - deccanherald.com",
            "The Telegraph India - telegraphindia.com",
            "Outlook India - outlookindia.com",
            "Frontline - frontline.thehindu.com",
            "Livemint - livemint.com",
            "Economic Times - economictimes.indiatimes.com",
            "Business Standard - business-standard.com",
            "Business Today - businesstoday.in",
            "Moneycontrol - moneycontrol.com",
            "The Wire - thewire.in",
            "Scroll.in - scroll.in",
            "The Print - theprint.in",
            "Alt News (India) - altnews.in",
            "BOOM Live (India) - boomlive.in",
            "Factly (India) - factly.in",
            "PolitiFact - politifact.com",
            "Snopes - snopes.com",
            "FactCheck.org - factcheck.org",
            "Full Fact (UK) - fullfact.org",
            "AFP Fact Check - factcheck.afp.com",
        ],
        description="Case-insensitive substrings matched against an item's `source` field and the domain in `url`. Fully user-configurable — swap in whatever outlets/gov domains matter for your corpus.",
    )
    source_credibility_known_score: float = Field(1.0, description="Score given when source/url matches an entry in reputable_sources")
    source_credibility_unknown_score: float = Field(0.5, description="Score given when source/url matches nothing on the allowlist — neutral-low rather than 0, since an unlisted source isn't necessarily illegitimate")


# ---------------------------------------------------------------------------
# 4. PARSING HELPERS
# ---------------------------------------------------------------------------
def _parse_date(raw: Optional[str]) -> Optional[datetime]:
    """Best-effort parse of a free-form date string into a tz-aware datetime.
    Returns None (rather than raising) for empty/placeholder/unparseable
    values, so callers can fall back to a neutral score."""
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or text.lower() in {"n/a", "na", "none", "not specified", "unknown", "-"}:
        return None
    if _HAS_DATEUTIL:
        try:
            dt = _dateutil_parser.parse(text, fuzzy=True)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, OverflowError, TypeError):
            return None
    # Minimal fallback if python-dateutil isn't installed.
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d %B %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


_AMOUNT_MULTIPLIERS = {
    "k": 1e3, "thousand": 1e3,
    "lakh": 1e5, "lac": 1e5,
    "crore": 1e7, "cr": 1e7,
    "m": 1e6, "million": 1e6, "mn": 1e6,
    "b": 1e9, "billion": 1e9, "bn": 1e9,
}
# Alternatives are ordered longest/most-specific first so e.g. "million"
# matches before the bare "m" fallback gets a chance to.
_AMOUNT_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<mult>lakh|lac|crore|cr|thousand|million|mn|billion|bn|k|m|b)?\b",
    re.IGNORECASE,
)
_CURRENCY_HINT_RE = re.compile(r"[$₹]|\brs\.?\b|\binr\b|\busd\b|,", re.IGNORECASE)
_BARE_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _parse_amount(raw: Optional[str]) -> Optional[float]:
    """
    Best-effort extraction of a numeric magnitude from a free-form
    `funding_amount` string, e.g. "$50,000", "Rs. 5 Lakh", "up to ₹2 Crore".

    Returns None when no plausible amount is found — including a guard
    against mistaking a bare year for a monetary figure, unless there's an
    actual currency/multiplier/comma cue. This is a heuristic for relative
    within-batch ranking, not a financial-reporting-grade normaliser.
    """
    if not raw or not isinstance(raw, str):
        return None
    stripped = raw.replace(",", "")
    match = _AMOUNT_RE.search(stripped)
    if not match or not match.group("num"):
        return None

    num_str = match.group("num")
    mult_key = (match.group("mult") or "").lower().rstrip(".")
    has_currency_hint = bool(_CURRENCY_HINT_RE.search(raw))

    if not mult_key and not has_currency_hint and _BARE_YEAR_RE.fullmatch(num_str):
        return None  # looks like a lone year, not an amount

    try:
        num = float(num_str)
    except ValueError:
        return None

    multiplier = _AMOUNT_MULTIPLIERS.get(mult_key, 1.0)
    return num * multiplier


_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "of", "in", "on", "to", "with", "is", "are",
    "be", "by", "at", "from", "this", "that", "any", "all", "who", "can", "apply",
}


def _tokenize(text: str) -> set:
    """Lowercase, strip punctuation, drop stopwords/very short tokens.
    Deliberately crude (no stemming/lemmatisation) — this backs a cheap,
    auditable overlap heuristic, not an NLP pipeline."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _grant_eligibility_match_score(eligibility: Optional[str], query: str) -> Optional[float]:
    """
    Fraction of the *query's* distinguishing terms that are echoed in the
    grant's `eligibility` text. Framed query-token-coverage-first (rather
    than symmetric Jaccard) because a user asking "grants for women-led
    agritech startups" wants to know how much of *that description* the
    eligibility criteria satisfy — not how much of the (often much longer)
    eligibility paragraph happens to relate to the query.

    Returns None (not 0.0) when either side has no usable tokens, so the
    signal is skipped rather than treated as "no match" — an item with no
    eligibility text extracted at all shouldn't be penalised twice (once by
    `completeness`, again here).
    """
    if not eligibility:
        return None
    e_tokens = _tokenize(eligibility)
    q_tokens = _tokenize(query)
    if not e_tokens or not q_tokens:
        return None
    return len(e_tokens & q_tokens) / len(q_tokens)


def _grant_indian_ngo_relevance_score(item: GrantItem, query_relevance: float) -> float:
    """Return an auditable Indian-NGO relevance score for a grant.

    The semantic/lexical query score is combined with explicit India and
    NGO/social-impact evidence from the extracted grant fields. This keeps
    the boolean grant decision explainable instead of treating an arbitrary
    funding amount or recency value as proof of relevance.
    """
    text = " ".join(
        value for value in [item.title, item.description, item.eligibility, item.application_url]
        if value
    ).lower()
    india_terms = ("india", "indian", ".in/", ".gov.in", "₹", "rs.", "inr")
    ngo_terms = (
        "ngo", "nonprofit", "non-profit", "non governmental", "non-governmental",
        "civil society", "social impact", "charity", "voluntary organisation", "voluntary organization",
    )
    india_evidence = 1.0 if any(term in text for term in india_terms) else 0.0
    ngo_evidence = 1.0 if any(term in text for term in ngo_terms) else 0.0
    return 0.45 * query_relevance + 0.30 * india_evidence + 0.25 * ngo_evidence


def _news_indian_ngo_relevance_score(item: NewsItem, query_relevance: float) -> float:
    """Score whether a news item concerns Indian NGOs/social impact."""
    text = " ".join(value for value in [item.title, item.summary, item.source, item.url] if value).lower()
    india_terms = ("india", "indian", ".in/", ".gov.in")
    ngo_terms = (
        "ngo", "nonprofit", "non-profit", "non governmental", "non-governmental",
        "civil society", "social impact", "csr", "charity", "voluntary organisation", "voluntary organization",
    )
    india_evidence = 1.0 if any(term in text for term in india_terms) else 0.0
    ngo_evidence = 1.0 if any(term in text for term in ngo_terms) else 0.0
    return 0.50 * query_relevance + 0.30 * india_evidence + 0.20 * ngo_evidence


def _source_credibility_score(
    source: Optional[str],
    url: Optional[str],
    reputable_sources: List[str],
    known_score: float,
    unknown_score: float,
) -> Optional[float]:
    """
    Checks `source` (and, failing that, the domain parsed out of `url`)
    against a configurable allowlist. This is a fast, fully-auditable proxy
    for "is this from known coverage or an unverified blog/press-release
    mirror" — not a trained classifier, and deliberately not one: an
    allowlist is transparent, instantly editable per corpus, and doesn't
    need training data.

    Returns None only when there is truly nothing to check (no source name
    and no URL were extracted at all), so the signal is skipped rather than
    treated as low-credibility — that gap is already captured by
    `completeness`.
    """
    haystack_parts = []
    if source:
        haystack_parts.append(source.lower())
    if url:
        try:
            netloc = urlparse(url).netloc.lower()
        except ValueError:
            netloc = ""
        haystack_parts.append(netloc or url.lower())
    if not haystack_parts:
        return None
    haystack = " ".join(haystack_parts)
    for domain in reputable_sources:
        if domain.lower() in haystack:
            return known_score
    return unknown_score


# ---------------------------------------------------------------------------
# 5. NORMALISATION / SCORING HELPERS
# ---------------------------------------------------------------------------
def _minmax_normalize(values: List[Optional[float]], default: float = 0.5) -> List[float]:
    """
    Scale raw scores to [0, 1] via min-max normalisation, corpus-relative
    (meaningful for ranking *within this batch*, not as an absolute score).

    `None` entries (signal unavailable for that item) receive `default` — a
    neutral midpoint — rather than 0, so a missing signal doesn't
    automatically sink an item; it's simply excluded from discriminating
    that item's score.
    """
    known = [v for v in values if v is not None]
    if not known:
        return [default for _ in values]
    lo, hi = min(known), max(known)
    if math.isclose(hi, lo):
        # Every candidate ties on this signal -> it carries no
        # discriminative information; treat everyone as "average".
        return [default if v is None else 1.0 for v in values]
    return [default if v is None else (v - lo) / (hi - lo) for v in values]


def _recency_score(pub_date: Optional[datetime], now: datetime, half_life_days: float, default: float) -> float:
    """
    Exponential-decay freshness score: 1.0 "published now", 0.5 at exactly
    `half_life_days` old, asymptotically -> 0 for old items.

    Exponential (half-life) decay is used instead of a hard cutoff or linear
    ramp because relevance of news/open schemes fades gradually rather than
    falling off a cliff, and a half-life gives one intuitive tuning knob per
    category (see `grant_recency_half_life_days` / `news_recency_half_life_days`).
    """
    if pub_date is None:
        return default
    age_days = max((now - pub_date).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (age_days / half_life_days)


def _grant_deadline_score(
    end_date: Optional[datetime],
    now: datetime,
    min_days_needed: float,
    ideal_days_remaining: float,
    days_std: float,
) -> Optional[float]:
    """
    Scores how "actionable" a grant's deadline is, in [0, 1].

    Favours deadlines that are: not already passed (a separate boolean grant
    decision rejects expired grants), not so close there's no time to prepare a real
    application (`min_days_needed`), and centred on a configurable "sweet
    spot" (`ideal_days_remaining`, default ~3 weeks out).

    Modelled as a one-sided Gaussian bump around `ideal_days_remaining`
    (spread `days_std`), gated to 0 below `min_days_needed`. Intentionally a
    simple, fully-tunable heuristic rather than anything learned.
    """
    if end_date is None:
        return None
    days_remaining = (end_date - now).total_seconds() / 86400.0
    if days_remaining < min_days_needed:
        return 0.0
    return math.exp(-((days_remaining - ideal_days_remaining) ** 2) / (2 * days_std ** 2))


def _grant_amount_scores(raw_values: List[Optional[str]], higher_is_better: bool) -> List[Optional[float]]:
    """
    Converts free-form `funding_amount` strings into a [0, 1] "grant size"
    score, log10-scaled and min-max normalised *within the current
    candidate batch* (a "large grant" is a corpus-relative notion, not a
    fixed global scale).

    Log-scaling matters because grant sizes commonly span orders of
    magnitude (e.g. a few thousand for a student stipend vs. several crore
    for an infrastructure grant); without it, a handful of huge grants would
    compress every smaller-but-still-relevant grant's score down near 0.

    Returns None (not 0.0) for entries with no parseable amount, so they can
    be excluded from this signal entirely rather than penalised for it.
    """
    amounts = [_parse_amount(v) for v in raw_values]
    log_amounts = [math.log10(a + 1.0) if (a is not None and a > 0) else None for a in amounts]
    normalized = _minmax_normalize(log_amounts, default=0.5)
    scores: List[Optional[float]] = []
    for original, norm in zip(log_amounts, normalized):
        if original is None:
            scores.append(None)
        else:
            scores.append(norm if higher_is_better else 1.0 - norm)
    return scores


def _ranks_desc(values: List[float]) -> List[int]:
    """1-indexed rank of each value within `values`, highest value = rank 1. Used by RRF fusion."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [0] * len(values)
    for position, idx in enumerate(order):
        ranks[idx] = position + 1
    return ranks


def _subset_ranks(indices: List[int], scores: List[Optional[float]]) -> Dict[int, int]:
    """Rank (1 = best) of each index in `indices`, computed only among the
    entries in `indices` that have a non-None score in `scores`. Entries
    with None are simply absent from the returned mapping (excluded from
    that RRF term entirely, the RRF analogue of the weighted-sum
    renormalisation trick)."""
    scored = [(i, scores[i]) for i in indices if scores[i] is not None]
    if not scored:
        return {}
    ordered = sorted(scored, key=lambda pair: pair[1], reverse=True)
    return {idx: pos + 1 for pos, (idx, _) in enumerate(ordered)}


# ---------------------------------------------------------------------------
# 6. THE RERANKER NODE
# ---------------------------------------------------------------------------
GRANT_OPTIONAL_FIELDS = ["description", "funding_amount", "eligibility", "start_date", "end_date", "application_mode", "application_url"]
NEWS_OPTIONAL_FIELDS = ["summary", "published_date", "source", "url"]


class RerankerNode:
    """
    Reranks the item list from the extraction node's ExtractionResult against a
    query, using category-aware signal sets described in the module
    docstring.

    Usage:
        node = RerankerNode(RerankConfig(top_fraction=0.1))
        result = node.rerank(extraction_result, query="renewable energy grants")
    """

    def __init__(self, config: Optional[RerankConfig] = None):
        self.config = config or RerankConfig()
        self._bi_encoder = None     # lazy-loaded sentence_transformers.SentenceTransformer
        self._cross_encoder = None  # lazy-loaded sentence_transformers.CrossEncoder

    # -- lazy model loading, so `import rerank_node` alone never needs torch -
    def _resolve_model_location(self, model_name: str, cache_name: str) -> str:
        local_path = Path(self.config.model_cache_dir).expanduser() / cache_name
        if local_path.exists() and any(local_path.iterdir()):
            return str(local_path)
        if self.config.allow_remote_model_downloads:
            return model_name
        raise FileNotFoundError(
            f"Predownloaded model not found at {local_path}. "
            "Run `python Rerank_Node/download_rerank_models.py` once, or pass "
            "`--allow-remote-model-downloads` to permit Hugging Face downloads at runtime."
        )

    def _get_bi_encoder(self):
        if self._bi_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for bi-encoder scoring. "
                    "Install with: pip install sentence-transformers"
                ) from e
            model_location = self._resolve_model_location(
                self.config.bi_encoder_model_name,
                BI_ENCODER_CACHE_NAME,
            )
            self._bi_encoder = SentenceTransformer(model_location, device=self.config.device)
        return self._bi_encoder

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder scoring. "
                    "Install with: pip install sentence-transformers"
                ) from e
            model_location = self._resolve_model_location(
                self.config.cross_encoder_model_name,
                CROSS_ENCODER_CACHE_NAME,
            )
            self._cross_encoder = CrossEncoder(model_location, device=self.config.device)
        return self._cross_encoder

    # -- category-aware helpers -----------------------------------------------
    @staticmethod
    def _item_text(item: ExtractedItem) -> str:
        """Text used for BM25/bi-encoder/cross-encoder scoring. Grants fold
        in `eligibility` too (it's often where the real matching signal
        lives, e.g. "women-led", "MSME", "Assam-based"); news items use
        `summary` in place of a grant's `description`."""
        if isinstance(item, GrantItem):
            parts = [item.title or "", item.description or "", item.eligibility or ""]
        else:
            parts = [item.title or "", item.summary or ""]
        return " ".join(p for p in parts if p).strip()

    @staticmethod
    def _item_completeness(item: ExtractedItem) -> float:
        """Fraction of the item's *own category's* optional fields that were
        filled in — a grant is graded against grant fields, a news item
        against news fields, not a shared generic list."""
        if isinstance(item, GrantItem):
            fields = GRANT_OPTIONAL_FIELDS
        else:
            fields = NEWS_OPTIONAL_FIELDS
        return sum(1 for f in fields if getattr(item, f, None)) / len(fields)

    def _item_recency(self, item: ExtractedItem, now: datetime) -> float:
        """Category-aware recency: grants decay off `start_date` on a slow
        clock, news decays off `published_date` on a fast one (see module
        docstring, 'RECENCY IS CATEGORY-AWARE')."""
        if isinstance(item, GrantItem):
            pub = _parse_date(item.start_date)
            half_life = self.config.grant_recency_half_life_days
        else:
            pub = _parse_date(item.published_date)
            half_life = self.config.news_recency_half_life_days
        return _recency_score(pub, now, half_life, self.config.recency_default_score)

    # -- individual shared scorers ---------------------------------------------
    def _bm25_scores(self, texts: List[str], query: str) -> List[float]:
        if not _HAS_BM25:
            query_tokens = _tokenize(query)
            if not query_tokens:
                return [0.5] * len(texts)
            raw = [
                len(query_tokens & _tokenize(text)) / len(query_tokens)
                for text in texts
            ]
            return _minmax_normalize(raw)
        tokenized_corpus = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        raw_scores = bm25.get_scores(query.lower().split())
        return _minmax_normalize(list(raw_scores))

    def _bi_encoder_scores(self, texts: List[str], query: str) -> List[float]:
        if not self.config.use_bi_encoder:
            return [0.5] * len(texts)
        try:
            model = self._get_bi_encoder()
            doc_embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            query_embedding = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        except Exception as e:
            print(f"[RERANK WARNING] Bi-encoder unavailable; using neutral scores. Error: {e}")
            return [0.5] * len(texts)
        # Embeddings are L2-normalised, so dot product == cosine similarity.
        # Computed with a plain loop (rather than `@`) so this also works
        # with any encoder that returns plain lists rather than numpy arrays.
        raw = [
            float(sum(a * b for a, b in zip(doc_emb, query_embedding)))
            for doc_emb in doc_embeddings
        ]
        return [(s + 1.0) / 2.0 for s in raw]  # [-1,1] -> [0,1]

    def _cross_encoder_scores(self, texts: List[str], query: str, prefilter_mask: List[bool]) -> List[float]:
        if not self.config.use_cross_encoder:
            return [0.5] * len(texts)
        try:
            model = self._get_cross_encoder()
        except Exception as e:
            print(f"[RERANK WARNING] Cross-encoder unavailable; using neutral scores. Error: {e}")
            return [0.5] * len(texts)
        pairs = [(query, t) for t, keep in zip(texts, prefilter_mask) if keep]
        if not pairs:
            return [0.0] * len(texts)
        try:
            raw_scores = model.predict(pairs)
        except Exception as e:
            print(f"[RERANK WARNING] Cross-encoder prediction failed; using neutral scores. Error: {e}")
            return [0.5] * len(texts)
        sigmoid_scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]
        result: List[float] = []
        idx = 0
        for keep in prefilter_mask:
            if keep:
                result.append(sigmoid_scores[idx])
                idx += 1
            else:
                result.append(0.0)
        return result

    # -- selection helper --------------------------------------------------
    def _select_top(self, indices: List[int], final_scores: List[float]) -> List[int]:
        if not indices:
            return []
        order = sorted(indices, key=lambda i: final_scores[i], reverse=True)
        keep_count = self.config.top_k if self.config.top_k is not None else math.ceil(len(indices) * self.config.top_fraction)
        keep_count = max(min(self.config.min_items_returned, len(indices)), min(keep_count, len(indices)))
        return order[:keep_count]

    # -- main entry point -----------------------------------------------------
    def rerank_with_all(self, extraction_result, query: str) -> tuple[RerankedResult, RerankedResult, RerankedResult]:
        """Return ``(all_grants, all_news, final_items)`` from one scoring pass."""
        if isinstance(extraction_result, dict):
            extraction_result = ExtractionResult(**extraction_result)
        items = list(extraction_result.items)

        n = len(items)
        if n == 0:
            empty_result = RerankedResult(
                query=query, items=[], total_candidates=0, returned_count=0,
                top_fraction=self.config.top_fraction, fusion_method=self.config.fusion_method,
                grant_weights=self.config.grant_weights, news_weights=self.config.news_weights,
            )
            return empty_result, empty_result, empty_result

        now = datetime.now(timezone.utc)
        texts = [self._item_text(it) for it in items]
        is_grant = [isinstance(it, GrantItem) for it in items]

        # ---- shared lexical + semantic signals ----
        bm25 = self._bm25_scores(texts, query)
        bi_enc = self._bi_encoder_scores(texts, query)

        # Cascade optimisation: only spend cross-encoder compute on the top
        # fraction by the cheap BM25+bi-encoder combo. Identical to scoring
        # everything when cross_encoder_prefilter_fraction == 1.0 (default).
        combo = [(b + e) / 2.0 for b, e in zip(bm25, bi_enc)]
        prefilter_count = max(1, math.ceil(n * self.config.cross_encoder_prefilter_fraction))
        keep_idx = set(sorted(range(n), key=lambda i: combo[i], reverse=True)[:prefilter_count])
        prefilter_mask = [i in keep_idx for i in range(n)]
        cross_enc = self._cross_encoder_scores(texts, query, prefilter_mask)

        # ---- shared, category-aware signals ----
        recency = [self._item_recency(it, now) for it in items]
        completeness = [self._item_completeness(it) for it in items]

        # ---- grant-exclusive signals ----
        end_dates = [_parse_date(it.end_date) if isinstance(it, GrantItem) else None for it in items]
        deadline_scores = [
            _grant_deadline_score(d, now, self.config.grant_min_days_needed,
                                   self.config.grant_ideal_days_remaining, self.config.grant_days_std)
            if g else None
            for d, g in zip(end_dates, is_grant)
        ]
        amount_scores = _grant_amount_scores(
            [it.funding_amount if isinstance(it, GrantItem) else None for it in items],
            self.config.higher_grant_amount_is_better,
        )
        eligibility_scores = [
            _grant_eligibility_match_score(it.eligibility, query) if isinstance(it, GrantItem) else None
            for it in items
        ]

        # ---- news-exclusive signals ----
        credibility_scores = [
            _source_credibility_score(
                it.source, it.url, self.config.reputable_sources,
                self.config.source_credibility_known_score, self.config.source_credibility_unknown_score,
            ) if isinstance(it, NewsItem) else None
            for it in items
        ]
        query_relevance = [(bm25[i] + bi_enc[i] + cross_enc[i]) / 3.0 for i in range(n)]
        grant_relevance_scores = [
            _grant_indian_ngo_relevance_score(it, query_relevance[i]) if isinstance(it, GrantItem) else None
            for i, it in enumerate(items)
        ]
        news_relevance_scores = [
            _news_indian_ngo_relevance_score(it, query_relevance[i]) if isinstance(it, NewsItem) else None
            for i, it in enumerate(items)
        ]
        grant_deadline_passed = [
            (end_dates[i] is not None and end_dates[i] < now) if is_grant[i] else None
            for i in range(n)
        ]
        grant_is_accepted = [
            bool(
                is_grant[i]
                and grant_relevance_scores[i] is not None
                and grant_relevance_scores[i] >= self.config.grant_indian_ngo_relevance_threshold
                and not grant_deadline_passed[i]
            )
            for i in range(n)
        ]

        breakdowns: List[ScoreBreakdown] = [
            ScoreBreakdown(
                bm25=bm25[i], bi_encoder=bi_enc[i], cross_encoder=cross_enc[i],
                recency=recency[i], completeness=completeness[i],
                grant_deadline=deadline_scores[i], grant_amount=amount_scores[i],
                grant_eligibility_match=eligibility_scores[i],
                grant_indian_ngo_relevance=grant_relevance_scores[i],
                grant_is_indian_ngo_relevant=(
                    grant_relevance_scores[i] >= self.config.grant_indian_ngo_relevance_threshold
                    if grant_relevance_scores[i] is not None else None
                ),
                grant_deadline_passed=grant_deadline_passed[i],
                source_credibility=credibility_scores[i],
                news_indian_ngo_relevance=news_relevance_scores[i],
            )
            for i in range(n)
        ]

        # ---- fuse (per item, using that item's own category's weight profile) ----
        final_scores: List[float]

        if self.config.fusion_method == "weighted_sum":
            final_scores = []
            for i in range(n):
                weights = self.config.grant_weights if is_grant[i] else self.config.news_weights
                components = {
                    "bm25": bm25[i], "bi_encoder": bi_enc[i], "cross_encoder": cross_enc[i],
                    "recency": recency[i], "completeness": completeness[i],
                }
                if is_grant[i]:
                    extra = {
                        "grant_deadline": deadline_scores[i],
                        "grant_amount": amount_scores[i],
                        "grant_eligibility_match": eligibility_scores[i],
                    }
                else:
                    extra = {
                        "source_credibility": credibility_scores[i],
                        "news_indian_ngo_relevance": news_relevance_scores[i],
                    }

                total_w = sum(weights.get(k, 0.0) for k in components)
                score = sum(components[k] * weights.get(k, 0.0) for k in components)
                for k, v in extra.items():
                    if v is not None:
                        score += v * weights.get(k, 0.0)
                        total_w += weights.get(k, 0.0)

                # Renormalise by the weight actually applied, so an item
                # missing a category-exclusive sub-signal (e.g. no
                # parseable eligibility text) isn't penalised for an
                # extraction gap rather than its actual relevance.
                final_scores.append(score / total_w if total_w > 0 else 0.0)

        elif self.config.fusion_method == "rrf":
            # Reciprocal Rank Fusion: score = sum(weight / (k + rank)) across
            # each signal's own ranking. Scale-free (only relative order
            # within each signal matters) — robust when component score
            # distributions differ wildly (cross-encoder logits vs. BM25 vs.
            # a Gaussian deadline bump) and you'd rather not hand-tune
            # relative magnitudes.
            shared_signals = {"bm25": bm25, "bi_encoder": bi_enc, "cross_encoder": cross_enc,
                               "recency": recency, "completeness": completeness}
            shared_rank_maps = {k: _ranks_desc(v) for k, v in shared_signals.items()}
            k_const = self.config.rrf_k

            grant_idxs = [i for i in range(n) if is_grant[i]]
            news_idxs = [i for i in range(n) if not is_grant[i]]
            deadline_rank = _subset_ranks(grant_idxs, deadline_scores)
            amount_rank = _subset_ranks(grant_idxs, amount_scores)
            eligibility_rank = _subset_ranks(grant_idxs, eligibility_scores)
            credibility_rank = _subset_ranks(news_idxs, credibility_scores)
            news_relevance_rank = _subset_ranks(news_idxs, news_relevance_scores)

            raw_rrf: List[float] = []
            for i in range(n):
                weights = self.config.grant_weights if is_grant[i] else self.config.news_weights
                score = sum(weights.get(k, 0.0) / (k_const + shared_rank_maps[k][i]) for k in shared_signals)
                if is_grant[i]:
                    if i in deadline_rank:
                        score += weights.get("grant_deadline", 0.0) / (k_const + deadline_rank[i])
                    if i in amount_rank:
                        score += weights.get("grant_amount", 0.0) / (k_const + amount_rank[i])
                    if i in eligibility_rank:
                        score += weights.get("grant_eligibility_match", 0.0) / (k_const + eligibility_rank[i])
                else:
                    if i in credibility_rank:
                        score += weights.get("source_credibility", 0.0) / (k_const + credibility_rank[i])
                    if i in news_relevance_rank:
                        score += weights.get("news_indian_ngo_relevance", 0.0) / (k_const + news_relevance_rank[i])
                raw_rrf.append(score)
            # RRF scores aren't naturally in [0,1]; normalise for a
            # consistent output range regardless of fusion_method.
            final_scores = _minmax_normalize(raw_rrf)
        else:
            raise ValueError(f"Unknown fusion_method: {self.config.fusion_method!r}")

        grant_indices = [i for i in range(n) if is_grant[i]]
        news_indices = [i for i in range(n) if not is_grant[i]]
        top_news_indices = self._select_top(news_indices, final_scores)
        accepted_grant_indices = [i for i in grant_indices if grant_is_accepted[i]]

        def _result_for(output_items: List[RerankedItem], total_candidates: int) -> RerankedResult:
            return RerankedResult(
                query=query,
                items=output_items,
                total_candidates=total_candidates,
                returned_count=len(output_items),
                top_fraction=self.config.top_fraction,
                fusion_method=self.config.fusion_method,
                grant_weights=self.config.grant_weights,
                news_weights=self.config.news_weights,
            )

        all_grants = [
            RerankedGrantItem(
                **items[i].model_dump(),
                final_score=grant_is_accepted[i],
                score_breakdown=breakdowns[i],
            )
            for i in grant_indices
        ]
        all_news = [
            RerankedNewsItem(
                **items[i].model_dump(),
                rank=rank + 1,
                final_score=round(final_scores[i], 6),
                score_breakdown=breakdowns[i],
            )
            for rank, i in enumerate(sorted(news_indices, key=lambda i: final_scores[i], reverse=True))
        ]
        final_items: List[RerankedItem] = [
            RerankedGrantItem(
                **items[i].model_dump(), final_score=True, score_breakdown=breakdowns[i]
            )
            for i in accepted_grant_indices
        ] + [
            RerankedNewsItem(
                **items[i].model_dump(), rank=rank + 1,
                final_score=round(final_scores[i], 6), score_breakdown=breakdowns[i]
            )
            for rank, i in enumerate(top_news_indices)
        ]
        return (
            _result_for(all_grants, len(grant_indices)),
            _result_for(all_news, len(news_indices)),
            _result_for(final_items, n),
        )

    def rerank(self, extraction_result, query: str) -> RerankedResult:
        """Return accepted grants plus the configured top slice of news."""
        _, _, final_items = self.rerank_with_all(extraction_result, query)
        return final_items


# ---------------------------------------------------------------------------
# 7. PIPELINE-NODE ADAPTER
# ---------------------------------------------------------------------------
def rerank_node(state: dict, query: Optional[str] = None, config: Optional[RerankConfig] = None) -> dict:
    """
    Thin adapter so this module can be dropped into a graph/pipeline (e.g. a
    LangGraph-style step) that threads a shared `state` dict between nodes.

    Expects  : state["extracted_data"] == the dict extraction_node.py produced
               (`{"items": [...]}`), and a query via the `query` arg or
               state["query"].
    Returns  : `state` with an added state["reranked_data"] key (the
               RerankedResult as a plain dict).
    """
    q = query or state.get("query")
    if not q:
        raise ValueError("A query is required to rerank against — pass `query=` or set state['query'].")
    node = RerankerNode(config)
    result = node.rerank(state["extracted_data"], q)
    state["reranked_data"] = result.model_dump()
    return state


# ---------------------------------------------------------------------------
# 8. BUILT-IN SYNTHETIC DATASET — for a dependency-light smoke test.
# ---------------------------------------------------------------------------
def _demo_data() -> dict:
    from datetime import timedelta
    today = datetime.now(timezone.utc)
    fmt = lambda dt: dt.strftime("%Y-%m-%d")
    items = [
        {
            "category": "grant",
            "title": "Green Startup Innovation Grant",
            "description": "Funding for early-stage startups building renewable energy hardware or software.",
            "funding_amount": "up to ₹25 Lakh",
            "eligibility": "Women-led or MSME-registered early-stage startups in the clean energy sector",
            "start_date": fmt(today - timedelta(days=10)),
            "end_date": fmt(today + timedelta(days=20)),
            "application_mode": "online",
            "application_url": "https://example.gov/apply/green-startup",
        },
        {
            "category": "grant",
            "title": "Solar Rooftop Subsidy Scheme",
            "description": "Subsidy for households and small businesses installing rooftop solar panels.",
            "funding_amount": "Rs. 40,000 per kW",
            "eligibility": "Households and small businesses in Assam",
            "start_date": fmt(today - timedelta(days=200)),
            "end_date": fmt(today + timedelta(days=2)),  # very soon — low actionability
            "application_mode": "offline",
            "application_url": None,
        },
        {
            "category": "grant",
            "title": "National Clean Energy Infrastructure Fund",
            "description": "Large-scale capital grant for clean energy infrastructure projects.",
            "funding_amount": "₹5 Crore",
            "eligibility": "Large enterprises and public sector undertakings",
            "start_date": fmt(today - timedelta(days=5)),
            "end_date": fmt(today - timedelta(days=1)),  # already expired -> boolean final_score is false
            "application_mode": "online",
            "application_url": "https://example.gov/apply/clean-energy-fund",
        },
        {
            "category": "news",
            "title": "City Council Approves New Wind Farm",
            "summary": "Local news coverage of a new wind energy project approved this week.",
            "published_date": fmt(today - timedelta(days=2)),
            "source": "Reuters",
            "url": "https://reuters.com/energy/wind-farm-approved",
        },
        {
            "category": "news",
            "title": "Startup Blog Claims Big Solar Breakthrough",
            "summary": "An unverified blog post about a claimed solar efficiency breakthrough, no named source.",
            "published_date": fmt(today - timedelta(days=1)),
            "source": "randomtechblog.example",
            "url": "https://randomtechblog.example/solar-breakthrough",
        },
        {
            "category": "news",
            "title": "Historical Retrospective: Energy Policy in the 1990s",
            "summary": "An old archived news piece with no current relevance.",
            "published_date": fmt(today - timedelta(days=900)),
            "source": "The Hindu",
            "url": "https://thehindu.com/archive/energy-policy-1990s",
        },
    ]
    return {"items": items}


# ---------------------------------------------------------------------------
# 9. COMMAND-LINE ENTRY POINT
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rerank extraction_node.py's JSON output with a hybrid, category-aware BM25 + bi-encoder + cross-encoder pipeline."
    )
    p.add_argument("input", nargs="?", default=None, help="Path to extracted_items.json produced by extraction_node.py")
    p.add_argument("--demo", action="store_true", help="Ignore `input` and rerank a small built-in synthetic dataset instead")
    p.add_argument("--query", default="renewable energy grants and funding opportunities for startups",
                   help="The query/interest to rank items against")
    p.add_argument("--top-fraction", type=float, default=0.10, help="Fraction of items to keep, e.g. 0.10 for top 10%%")
    p.add_argument("--top-k", type=int, default=None, help="Absolute number of items to keep (overrides --top-fraction)")
    p.add_argument("--per-category-top", action="store_true", help="Retained for compatibility; only news items use the top cutoff.")
    p.add_argument("--fusion-method", choices=["weighted_sum", "rrf"], default="weighted_sum")
    p.add_argument("--model-cache-dir", default=str(DEFAULT_MODEL_CACHE_DIR), help="Directory containing predownloaded reranker models")
    p.add_argument("--allow-remote-model-downloads", action="store_true", help="Allow runtime Hugging Face downloads if local models are missing")
    p.add_argument("--no-semantic-rerank", action="store_true", help="Skip bi/cross-encoder scoring entirely")
    p.add_argument("--output", default="reranked_items.json", help="Path for accepted grants plus the configured top-ranked news.")
    p.add_argument("--all-grants-output", default="scored_grant_items.json", help="Path for every grant and its boolean final score.")
    p.add_argument("--all-news-output", default="scored_news_items.json", help="Path for every news item and its continuous final score.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.demo:
        data = _demo_data()
    else:
        if not args.input:
            raise SystemExit("Provide an input JSON path, or pass --demo to run on a built-in synthetic dataset.")
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

    config = RerankConfig(
        top_fraction=args.top_fraction, top_k=args.top_k,
        per_category_top=args.per_category_top, fusion_method=args.fusion_method,
        model_cache_dir=args.model_cache_dir,
        allow_remote_model_downloads=args.allow_remote_model_downloads,
        use_bi_encoder=not args.no_semantic_rerank,
        use_cross_encoder=not args.no_semantic_rerank,
    )
    node = RerankerNode(config)
    all_grants, all_news, result = node.rerank_with_all(data, args.query)
    all_grants_payload = {"items": all_grants.model_dump()["items"]}
    all_news_payload = {"items": all_news.model_dump()["items"]}
    result_payload = {"items": result.model_dump()["items"]}

    with open(args.all_grants_output, "w", encoding="utf-8") as f:
        json.dump(all_grants_payload, f, indent=2, ensure_ascii=False)
    with open(args.all_news_output, "w", encoding="utf-8") as f:
        json.dump(all_news_payload, f, indent=2, ensure_ascii=False)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)

    print(f"Scored {result.total_candidates} candidate(s) -> kept {result.returned_count} final item(s) "
          f"(fusion={result.fusion_method}). Saved grants to {args.all_grants_output}, "
          f"news to {args.all_news_output}, and final results to {args.output}")
    print(json.dumps(result_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# POSSIBLE EXTENSIONS (not implemented, to keep this module focused)
# ---------------------------------------------------------------------------
# - Diversity / de-duplication (e.g. Maximal Marginal Relevance) as a final
#   pass over the kept set, so near-duplicate grants/news don't crowd out
#   the top slice at the expense of variety.
# - A learned fusion layer (e.g. logistic regression over the same signals)
#   once click/apply/read feedback is available, instead of hand-set weights.
# - A third category profile (e.g. "event") following the same pattern, if
#   extraction_node.py's schema grows a third member of the discriminated union.
