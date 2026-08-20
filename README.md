# Impact Weaver AI/ML Project Pipeline

This project is an automated discovery, extraction, reranking, and persistence
pipeline for identifying relevant grants, funding opportunities, and nonprofit
sector news for Indian NGOs.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Project Structure](#project-structure)
- [Ubuntu 22.04 Server Setup](#ubuntu-2204-server-setup)
- [Production Run Steps](#production-run-steps)
- [API Usage](#api-usage)
- [MongoDB Persistence](#mongodb-persistence)
- [Known Issues](#known-issues)
- [Future Scope](#future-scope)

## Problem Statement

The earlier website flow was posting unwanted and irrelevant articles because
the search and filtering logic was mainly based on keyword or word matching.
This approach was brittle: pages could match the right words but still be
generic, outdated, promotional, duplicated, or irrelevant to Indian NGOs.

The project needed a more sophisticated, automated, and LLM-assisted approach
that could:

- discover relevant links from live search results
- understand whether a page is a grant or a news item
- extract structured information from each page
- rerank the extracted items using relevance signals

## Solution Approach

### Workflow Diagram

![Final AI/ML Project Workflow](Final_AIML_Project_Workflow.drawio.png)

### Current Implemented Workflow

The current codebase implements the pipeline in the following stages:

1. **ReAct Discovery Node**

   File: `ReAct_Node/discovery_agent.py`

   The discovery node uses an LLM-powered ReAct style agent with live search
   tools. Instead of relying only on keyword matching, it searches for current
   grant opportunities, CSR funding opportunities, open calls, and nonprofit
   sector news. The node then asks an LLM parser to return a structured JSON
   object containing validated links with:

   - `title`
   - `url`
   - `type`: `grant` or `news`
   - `context_summary`
   - `publish_date`

2. **Extraction Node**
# Impact Weaver AI/ML Project Pipeline

This project is an automated discovery, extraction, reranking, and persistence
pipeline for identifying relevant grants, funding opportunities, and nonprofit
sector news for Indian NGOs.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Project Structure](#project-structure)
- [Ubuntu 22.04 Server Setup](#ubuntu-2204-server-setup)
- [Production Run Steps](#production-run-steps)
- [API Usage](#api-usage)
- [MongoDB Persistence](#mongodb-persistence)
- [Known Issues](#known-issues)
- [Future Scope](#future-scope)

## Problem Statement

The earlier website flow was posting unwanted and irrelevant articles because
the search and filtering logic was mainly based on keyword or word matching.
This approach was brittle: pages could match the right words but still be
generic, outdated, promotional, duplicated, or irrelevant to Indian NGOs.

The project needed a more sophisticated, automated, and LLM-assisted approach
that could:

- discover relevant links from live search results
- understand whether a page is a grant or a news item
- extract structured information from each page
- rerank the extracted items using relevance signals

## Solution Approach

### Workflow Diagram

![Final AI/ML Project Workflow](Final_AIML_Project_Workflow.drawio.png)

### Current Implemented Workflow

The current codebase implements the pipeline in the following stages:

1. **ReAct Discovery Node**

   File: `ReAct_Node/discovery_agent.py`

   The discovery node uses an LLM-powered ReAct style agent with live search
   tools. Instead of relying only on keyword matching, it searches for current
   grant opportunities, CSR funding opportunities, open calls, and nonprofit
   sector news. The node then asks an LLM parser to return a structured JSON
   object containing validated links with:

   - `title`
   - `url`
   - `type`: `grant` or `news`
   - `context_summary`
   - `publish_date`

2. **Extraction Node**

   File: `Extraction_Node/extraction_node.py`

   The extraction node uses Crawl4AI and an LLM extraction strategy to process
   each discovered URL. It extracts one JSON object per URL.

   Grant URLs are extracted into the `GrantItem` schema, which includes fields
   such as title, description, funding amount, eligibility, dates, application
   mode, application URL, and `main_media_url` for the primary image.

   News URLs are extracted into the `NewsItem` schema, which includes title,
   summary, published date, publisher, URL, and `main_media_url` for the primary image.
   
   To ensure images are reliably extracted while keeping LLM token usage cost-effective, the extractor processes raw HTML but aggressively prunes non-content tags (like `<script>`, `<style>`, and `<nav>`).

3. **Rerank Node**

   File: `Rerank_Node/rerank_node.py`

   The rerank node scores extracted items using a hybrid ranking approach. It
   combines lexical matching, transformer-based semantic similarity, category
   specific recency, completeness, grant deadline signals, grant relevance, and
   news relevance signals.

   The reranker separates:

   - selected items,
   - all scored grant items,
   - all scored news items.

   API responses and MongoDB payloads are normalized to:

   ```json
   {
     "items": []
   }
   ```

4. **Pipeline Orchestration**

   File: `pipeline.py`

   The pipeline connects discovery, extraction, and reranking. For FastAPI
   calls, the pipeline runs in memory and does not store intermediate outputs
   locally. The CLI path can still save local JSON files for debugging or
   manual runs.

5. **MongoDB Persistence**

   File: `mongo_store.py`

   The final selected and not-selected outputs are saved to MongoDB Atlas.
   Each document is saved with metadata such as `pipeline_run_id`,
   `selection_status`, `saved_at`, and `saved_at_iso`.

6. **FastAPI Service**

   File: `api.py`

   FastAPI exposes endpoints to run the full pipeline and return selected
   items. It also supports a background pipeline mode with job-status polling.

## Project Structure

```text
Impact Weaver/
├── api.py
├── mongo_store.py
├── pipeline.py
├── requirements.txt
├── README.md
├── ReAct_Node/
│   ├── discovery_agent.py
│   ├── prompts.py
│   └── tools.py
├── Extraction_Node/
│   ├── extraction_node.py
└── Rerank_Node/
    ├── rerank_node.py
    ├── download_rerank_models.py
    └── model_cache/
```

## Ubuntu 22.04 Server Setup

### 1. Install System Packages

```bash
sudo apt update
sudo apt install -y git curl build-essential wget
```

### 2. Install Miniforge
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

bash Miniforge3-Linux-x86_64.sh

source ~/miniforge3/bin/activate
conda init bash
source ~/.bashrc
```

Verify Installation:
```bash
conda --version
```

### 3. Clone the Repository

```bash
git clone <repository-url> impact-weaver-news-and-grants
cd impact-weaver-news-and-grants
```

### 4. Create the Conda Environment

```bash
conda env create -f setup.yml
```

Activate it 

```bash
conda activate impact-weaver-news-and-grants
```

To update the environment after dependency changes:

```bash
conda env update -f setup.yml --prune
```

### 5. Install Crawl4AI and Browser Dependencies

```bash
crawl4ai-setup
python -m playwright install --with-deps chromium
```

### 6. Configure Environment Variables

Create a `.env` file in the project root:

```bash
nano .env
```

Required variables:

```env
LITELLM_API_KEY="your-litellm-api-key"
LITELLM_BASE_URL="https://llm.impactweaver.com"
MONGODB_URI="your-mongodb-atlas-uri"
```

Optional MongoDB variables:

```env
MONGODB_SELECTED_COLLECTION="ai_news_grants"
MONGODB_NOT_SELECTED_NEWS_COLLECTION="ai_not_selected_news"
MONGODB_NOT_SELECTED_GRANT_COLLECTION="ai_not_selected_grants"
```

Do not commit `.env` to git.

### 7. Predownload Reranker Models

Run this once so runtime execution can use the local model cache:

```bash
python Rerank_Node/download_rerank_models.py
```

The models are saved under:

```text
Rerank_Node/model_cache/
```

## Production Run Steps

### 1. Verify Server Health Locally

```bash
conda activate impact-weaver-news-and-grants
uvicorn api:app --host 0.0.0.0 --port 8000
```

Check:

```bash
curl -u username:admin http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

### 2. Run with systemd

Create a service file:

```bash
sudo nano /etc/systemd/system/impact-weaver-news-and-grants.service
```

Example service:

```ini
[Unit]
Description=Impact Weaver News and Grants FastAPI Service
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/impact-weaver-news-and-grants
EnvironmentFile=/opt/impact-weaver-news-and-grants/.env
ExecStart=/home/ubuntu/miniforge3/envs/impact-weaver-news-and-grants/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable impact-weaver-news-and-grants
sudo systemctl start impact-weaver-news-and-grants
sudo systemctl status impact-weaver-news-and-grants
```

View logs:

```bash
journalctl -u impact-weaver-news-and-grants -f
```

### 3. Optional Reverse Proxy

In production, place Nginx in front of Uvicorn and terminate HTTPS at Nginx or
at a load balancer. Keep the FastAPI service bound to the internal server or
private network where possible.

## API Usage

Interactive API documentation is available at:

```text
http://<server-ip>:8000/docs
```

### Authentication

All API endpoints are protected by HTTP Basic Authentication. You must provide valid credentials using the `Authorization: Basic <base64>` header.
The credentials **must** be configured via environment variables in your `.env` file:
```env
API_USERNAME=your_username
API_PASSWORD=your_password
```
*(Note: If these variables are not set in your `.env` file, all API requests will fail as there is no longer a default login.)*

When using `curl`, you can pass them via the `-u` flag.

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

### Run Pipeline and Return Selected Items

```http
POST /pipeline/run
```

Default request body:

```json
{}
```

Default response:

```json
{
  "items": []
}
```

Example curl:

```bash
curl -X POST "http://127.0.0.1:8000/pipeline/run" \
  -H "Content-Type: application/json" \
  -u username:admin \
  -d '{}'
```

When using Swagger UI at `/docs`, do not submit the generated placeholder
values such as `"master_query": "string"` or `"query": "string"`. Either send
`{}` for the default full run, or set those fields to `null` unless you are
intentionally providing a custom search/reranking query. The API treats blank
strings and the Swagger `"string"` placeholder as omitted.

### Run Pipeline with Metadata and Not Selected Items

```bash
curl -X POST "http://127.0.0.1:8000/pipeline/run" \
  -H "Content-Type: application/json" \
  -u admin:admin \
  -d '{
    "include_metadata": true,
    "include_not_selected": true
  }'
```

Response shape:

```json
{
  "pipeline_run_id": "string",
  "selected": {
    "items": []
  },
  "mongo_save": "scheduled",
  "not_selected": {
    "grant_items": [],
    "news_items": []
  }
}
```

### Run Pipeline as a Background Job

```http
POST /pipeline/run-background
```

Response:

```json
{
  "status": "queued",
  "pipeline_run_id": "string",
  "status_url": "/pipeline/jobs/<pipeline_run_id>"
}
```

*Note: Background run progress (including start, success/failure, and MongoDB persistence events) is actively logged to the server console (`sys.stderr`). If running via systemd, you can monitor these logs using `journalctl -u impact-weaver-news-and-grants -f`.*

Check job status:

```http
GET /pipeline/jobs/{pipeline_run_id}
```

### API Parameters

| Parameter | Type | Default | Description |
|---|---:|---|---|
| `master_query` | string or null | `null` | Optional custom query for the ReAct discovery node. |
| `query` | string or null | pipeline default | Optional reranking query. |
| `limit` | integer or null | `null` | Maximum number of discovered links to extract. Useful for testing. |
| `skip_agent` | boolean | `false` | Use saved discovery JSON instead of running live discovery. Mainly for local debugging. |
| `skip_extract` | boolean | `false` | Use saved extraction JSON and run only reranking. Mainly for local debugging. |
| `provider` | string | `litellm_proxy/lite` | LLM provider string used by the extraction node. |
| `input_format` | enum | `markdown` | Crawl4AI content format: `markdown`, `html`, or `fit_markdown`. **Note: The default is `markdown` to save tokens. If image extraction fails on certain sites, you can change this to `html` directly in the FastAPI Swagger body for 100% extraction accuracy.** |
| `chunk_token_threshold` | integer | `1200` | Maximum token size per extraction chunk. |
| `overlap_rate` | float | `0.1` | Chunk overlap ratio from `0.0` to `1.0`. |
| `no_chunking` | boolean | `false` | Disable automatic chunking. |
| `temperature` | float | `0.0` | LLM extraction temperature. |
| `max_tokens` | integer | `1500` | Maximum LLM output tokens for extraction. |
| `top_fraction` | float | `0.10` | Fraction of news candidates to keep during reranking. |
| `top_k` | integer or null | `null` | Absolute number of news items to keep. Overrides `top_fraction` when set. |
| `per_category_top` | boolean | `false` | Apply top cutoff separately to grants and news. |
| `fusion_method` | enum | `weighted_sum` | Rerank fusion method: `weighted_sum` or `rrf`. |
| `no_semantic_rerank` | boolean | `false` | Disable sentence-transformer semantic reranking. |
| `allow_remote_model_downloads` | boolean | `false` | Permit Hugging Face model downloads at runtime if cache is missing. |
| `save_to_mongo` | boolean | `true` | Persist selected and not-selected outputs to MongoDB Atlas. |
| `include_not_selected` | boolean | `false` | Include not-selected grant and news items in the API response. |
| `include_metadata` | boolean | `false` | Include `pipeline_run_id` and Mongo save status in the response. |

## MongoDB Persistence

The API stores items in three MongoDB Atlas collections:

- `ai_news_grants`
- `ai_not_selected_news`
- `ai_not_selected_grants`

Each saved document includes:

- the original item fields,
- `pipeline_run_id`,
- `selection_status`,
- `saved_at`,
- `saved_at_iso`.

Before insertion, each document is validated with the MongoDB persistence
schema in `mongo_store.py`. Grant documents require `category: "grant"` and a
boolean `final_score`; news documents require `category: "news"` and a numeric
`final_score`. Both document types require `title`, `score_breakdown`, and the
MongoDB metadata fields listed above. Extra item fields are preserved so future
reranking signals can be stored without changing the database writer.

The selected collection contains the items returned by the reranker. The not
selected collections contain scored grants and news items that were processed
but not included in the selected output.

## Known Issues

1. **ReAct node can fail due to LLM behavior**

   The discovery node depends on LLM reasoning and JSON parsing. Sometimes the
   LLM may return incomplete JSON, malformed JSON, unexpected fields, or text
   around the JSON object. This can cause the ReAct node to fail before
   extraction starts.

2. **Live search results are non-deterministic**

   Search results can change between runs. A URL that appears in one run may
   disappear or be ranked differently in another run.

3. **Extraction can fail on difficult websites**

   Some websites block automated browsers, load content dynamically, or return
   low-quality page content to crawlers. In those cases the extractor may fall
   back to partial data.

4. **Runtime models require local cache or network access**

   The reranker expects sentence-transformer models in the local cache unless
   `allow_remote_model_downloads` is enabled. Production should predownload
   the models.

5. **Background job status is in memory**

   The `/pipeline/run-background` job status is stored in process memory. If
   the API process restarts, old job status records are lost.

## Future Scope

1. **LLM-based reranking**

   Replace or augment the current transformer-based reranking with an
   LLM-based judgment layer. This would allow the system to reason more deeply
   about grant eligibility, NGO relevance, policy relevance, and actionability.

2. **More robust ReAct output validation**

   Add automatic retries, schema repair, stricter JSON extraction, and fallback
   prompts when the discovery LLM returns invalid JSON.

3. **Persistent background job tracking**

   Store background job status in MongoDB or Redis instead of memory so job
   progress survives process restarts.

4. **Queue-based production execution**

   Move long-running pipeline execution to Celery, RQ, or another queue system.
   FastAPI would enqueue jobs while workers perform crawling, extraction, and
   reranking.

5. **Improved deduplication**

   Add URL canonicalization, title similarity, and content similarity checks to
   reduce duplicate grants and repeated news articles.

6. **Feedback-driven ranking**

   Capture user clicks, saves, rejects, and edits to learn better ranking
   weights over time.

7. **Monitoring and observability**

   Add structured logs, run metrics, token usage tracking, model latency,
   failure counts, and alerting for production operations.

8. **Admin review dashboard**

   Build an internal review UI where selected and not-selected items can be
   inspected, corrected, approved, or rejected before publishing.

9. **Better source trust scoring**

   Expand the source credibility logic with a maintained allowlist, blocklist,
   publisher reputation score, and government-domain prioritization.

