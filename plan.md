aku ingin membuat sistem pembuatan novel dengan multi agent  memakai fokus core pada crewai, untuk model agent pakai openrouter dan gemini, strukturnya:

novel-ai-system/
├── .env.example
├── README.md
├── docker-compose.yml
├── requirements.txt
├── scripts/                         # helper scripts (ingest, build-embeddings)
│   ├── ingest.py
│   └── embed_batch.py
│
├── backend/                         # API layer (FastAPI)
│   ├── Dockerfile
│   ├── app/
│   │   ├── main.py                  # FastAPI app entry (uvicorn backend.app.main:app)
│   │   ├── config.py                # centralized settings (load .env)
│   │   ├── routes/
│   │   │   ├── generate.py          # POST /generate_chapter
│   │   │   ├── job.py               # GET /job_status/{job_id}
│   │   │   └── qa.py                # POST /qa_review
│   │   ├── services/
│   │   │   ├── orchestrator.py      # enqueue job, job lifecycle helpers
│   │   │   └── api_clients.py       # wrappers: qdrant, mongo, neo4j, redis, embed API
│   │   ├── schemas/                 # Pydantic API contract (requests/responses)
│   │   │   ├── requests.py
│   │   │   └── responses.py
│   │   └── db/
│   │       ├── mongodb_client.py
│   │       ├── neo4j_client.py
│   │       └── redis_client.py
│   └── requirements.txt
│
├── workers/                         # Background workers / task runners
│   ├── Dockerfile
│   ├── worker.py                    # worker process (RQ/Celery wrapper)
│   └── tasks/
│       └── crew_tasks.py            # run_generate_job(job_id, payload) -> orchestrator
│
├── agents/                          # CrewAI & agent implementations (Python)
│   ├── novel_crew.py                # CrewAI crew definition (optional YAML binding)
│   ├── crew_configs.py              # Python config for agents (preferred)
│   └── tools/
│       ├── novel_tools.py           # helpers used by agents (save, fetch)
│       ├── qa_tools.py              # helpers for QA parsing/JSON schema
│       └── context_tools.py         # wrappers to query Qdrant/Neo4j
│
├── qa_system/                       # Multi-layer QA modules
│   ├── structural_qa.py
│   ├── character_qa.py
│   ├── style_qa.py
│   └── technical_qa.py
│
├── embeddings/                      # Chunking + embedding + vector ingest
│   ├── chunker.py                   # tiktoken + sentence splitter
│   ├── embedder.py                  # e5-large-v2 wrapper (or HF API)
│   ├── qdrant_client.py             # upsert/search helpers
│   └── retriever.py                 # LangChain wrapper (optional)
│
├── db/                              # Local DB / schema helpers (migrations if needed)
│   ├── mongo/                       # sample init scripts (optional)
│   ├── neo4j/                       # cypher init scripts (optional)
│   └── qdrant/                      # collection config scripts
│
├── configs/                         # optional YAML configs for hybrid mode
│   ├── agents.yaml                  # declarative agent roles (editable by non-dev)
│   └── tasks.yaml
│
├── infra/                           # infra + docker compose + deploy manifests
│   ├── docker-compose.yml
│   └── k8s/                         # optional k8s manifests
│
├── tests/                           # unit + integration tests
│   ├── test_chunker.py
│   ├── test_embedder.py
│   └── test_orchestrator.py
└── docs/
    └── api_contract.md              # human-friendly contract (copy from Pydantic/OpenAPI)
backend/app/

main.py — FastAPI app. Semua route (generate, job, qa) disini. FastAPI bantu auto-OpenAPI docs (docs/).

config.py — Satu sumber truth buat env (DB urls, API keys, model choice) — jangan sebar kunci di code.

routes/ — Endpoint HTTP publik. Backend hanya expose kontrak API (decoupling).

services/orchestrator.py — Jantung yang mem-push job ke Redis queue dan menulis job doc ke MongoDB.

api_clients.py — Abstraksi panggilan ke Qdrant, Mongo, Neo4j, embeddings provider. Biar gampang ganti provider tanpa touch route logic.

schemas/ — Pydantic models = kontrak API. Backend + Frontend pegang ini.

Kenapa: Backend bertugas validasi/auth/authorization + orchestrasi, bukan implementasi detail agent.

workers/

worker.py — process yang consume Redis queue (RQ/Celery). Menjalankan crew_tasks.run_generate_job.

tasks/crew_tasks.py — implement flow: retrieve context → writer agent → QA agents → decide revise → persist.

Kenapa: pekerja background buat kerja model yang berat, supaya API tetap responsif. Redis queue + worker pattern itu simpel & andal.

agents/ & configs/

novel_crew.py — definisi Crew (CrewAI) bila lo mau gunakan CrewAI CLI; crew_configs.py (Python) adalah cara preferensi gue untuk akurasi.

configs/agents.yaml — optional: human-editable declaration. Python loader akan bind YAML into agent constructors (hybrid).

tools/ — kode util yang agent panggil (DB access, prompt templates, aggregator).

Kenapa hybrid: Python-first + optional YAML = best of both worlds. Lo dapat akurasi & logika kuat, plus fleksibilitas non-dev tweak.

qa_system/ (multi-layer QA)

Tiap file implement spesialis QA: structural, character, style, technical.

Output JSON schema konsisten: {score:int, issues:[], patches:[]}.

Kenapa: memisahkan domain membuat checks lebih presisi & bisa paralelisasi. Urutan run: structural→character→style→technical.

embeddings/

chunker.py — pakai tiktoken + sentence splitter; parameter: chunk_size, overlap.

embedder.py — default intfloat/e5-large-v2 via sentence-transformers; menyediakan batching & normalization.

qdrant_client.py — ensure collection, upsert points, search wrapper.

retriever.py — (opsional) LangChain wrapper untuk convo/chain convenience.

Kenapa: memisahkan concerns => lebih mudah tuning embed model, batch size, dan mengganti vector DB.

db/

Skrip inisialisasi atau contoh cypher untuk Neo4j (character graph).

Qdrant collection setup (dimensions), MongoDB index setup.

Kenapa: siap production dan reproducible infra.

configs/agents.yaml (optional)

Contoh minimal:

writer:
  model: gpt-4o-mini
  role: "Long-form fiction writer"
  temperature: 0.7

structural_qa:
  model: gpt-4o-mini
  role: "Story continuity analyst"


Python dapat memuat ini jika ada; kalau tidak, gunakan agents/crew_configs.py.

Prioritas fitur Phase 1 (urutan implementasi)

Infra dev: docker-compose (qdrant, redis, mongo)

Chunking + Embedding ingest (scripts/ingest.py) — buat index worldbook/sample novel ke Qdrant

Backend API (FastAPI): POST /generate/chapter, GET /job/{id}

Job queue (Redis + RQ) + worker (workers/worker.py) + job lifecycle persisting ke MongoDB

GeneratorAgent minimal (wrapper LLM call)

TechnicalQAAgent (grammar + formatting, JSON output)

Orchestrator: run_generate_job implement flow + revise loop (max 2)

Persistence: chapter_versions in MongoDB, jobs collection

Logging/metrics basic (job durations, QA score)

Tests: unit test chunker + embedder + integration smoke test (end-to-end local)

Struktur file minimal Phase 1 (yang harus lo buat sekarang)
novel-ai-system/
├── backend/
│   └── app/
│       ├── main.py
│       ├── config.py
│       ├── routes/generate.py
│       ├── routes/job.py
│       ├── services/orchestrator.py
│       ├── schemas/requests.py
│       ├── schemas/responses.py
│       └── db/mongodb_client.py
├── workers/
│   ├── worker.py
│   └── tasks/generate_task.py
├── embeddings/
│   ├── chunker.py
│   ├── embedder.py
│   └── qdrant_client.py
├── agents/
│   ├── generator_agent.py
│   └── technical_qa.py
├── scripts/
│   └── ingest.py
└── infra/
    └── docker-compose.yml

Detail implementasi langkah-per-langkah
1) Infra dev — infra/docker-compose.yml

Minimal services: qdrant, redis, mongo, backend, worker.
Contoh (singkat):

version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:v1.11.0
    ports: ["6333:6333"]
  redis:
    image: redis:7
    ports: ["6379:6379"]
  mongo:
    image: mongo:6
    ports: ["27017:27017"]
  backend:
    build: ./backend
    ports: ["8000:8000"]
    depends_on: [qdrant, redis, mongo]
  worker:
    build: ./workers
    depends_on: [qdrant, redis, mongo]


Acceptance: docker compose up → all containers healthy.

2) Chunker — embeddings/chunker.py

Gunakan tiktoken + sentence splitting. Parameter praktis:

MAX_TOKENS = 800

OVERLAP = 64
Snippet:

import re, tiktoken
ENC = "cl100k_base"
def chunk_text(text, max_tokens=800, overlap=64):
    enc = tiktoken.get_encoding(ENC)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks=[]; cur=[]; cur_t=0
    for s in sentences:
        tks = len(enc.encode(s))
        if cur_t + tks <= max_tokens:
            cur.append(s); cur_t += tks
        else:
            chunks.append(" ".join(cur))
            cur = cur[-1:] if overlap>0 else []
            cur.append(s)
            cur_t = sum(len(enc.encode(x)) for x in cur)
    if cur: chunks.append(" ".join(cur))
    return chunks


Acceptance: chunk length <= ~max_tokens and overlap present.

3) Embedder — embeddings/embedder.py

Default: intfloat/e5-large-v2 via sentence-transformers. Provide HF inference fallback.
Snippet:

from sentence_transformers import SentenceTransformer
MODEL="intfloat/e5-large-v2"
model = SentenceTransformer(MODEL)
def embed_texts(texts, batch=32):
    return model.encode(texts, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True)


Acceptance: embeddings shape (N,1024) and normed.

4) Qdrant client — embeddings/qdrant_client.py

Create collection, upsert points (id, vector, payload).

Collection name: novel_chunks

dim=1024, distance=Cosine
Acceptance: can upsert and search: search(query_vector, top=3) returns hits.

5) Ingest script — scripts/ingest.py

Flow: read worldbook.md → chunk_text → embed_texts → upsert to qdrant with metadata.
CLI: python scripts/ingest.py --file data/worldbook.md --project myproj
Acceptance: qdrant has points; langs: verify qdrant_client.search returns related chunks.

6) Backend API — FastAPI
Pydantic models

backend/app/schemas/requests.py

from pydantic import BaseModel
class GenerateRequest(BaseModel):
    user_id: str
    project_id: str
    chapter_id: str | None = None
    prompt: str
    settings: dict = {"length_words":1200,"max_revision_rounds":2}


routes/generate.py

@router.post("/chapter", status_code=202)
def generate_chapter(req: GenerateRequest):
    job_id = orchestrator.enqueue_job(req.dict())
    return {"status":"queued","code":202,"data":{"job_id":job_id}}


routes/job.py GET /job/{id} returns job status from Mongo jobs collection.
Acceptance: POST /generate/chapter returns job_id; GET /job/{id} returns state.

7) Job queue + worker

Use RQ (simple) or Celery. Minimal RQ example:

backend/services/orchestrator.py — create job doc in Mongo, push to Redis queue with payload.

workers/worker.py — RQ worker that listens & calls tasks/generate_task.run_generate_job(job_id, payload).

Job doc structure in Mongo jobs:

{
  "_id": ObjectId,
  "job_id": "uuid",
  "user_id":"u1",
  "project_id":"p1",
  "payload":{...},
  "state":"queued|running|success|failed|needs_human",
  "progress":0.0,
  "result": null,
  "created_at": ...
}


Acceptance: job doc created, worker picks job → state becomes running.

8) GeneratorAgent — agents/generator_agent.py

Wrap LLM calls. Minimal async wrapper for OpenAI (or local model):

import openai
async def generate_text(prompt, max_tokens=1000, temp=0.7):
    resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=max_tokens, temperature=temp)
    return resp.choices[0].message.content


Prompt template (Generator):

SYSTEM: You are a professional novelist. Tone: {tone}. Do not contradict context. 
USER: CONTEXT: {context}
Task: {prompt}
Constraints: max words ~{length_words}. If uncertain mark [VERIFY].
Output: Markdown chapter only.


Acceptance: returns human-readable chapter text.

9) Technical QA agent — agents/technical_qa.py

Goal: fast grammar & formatting fixes + JSON schema response.

Prompt template (strict JSON):

SYSTEM: You are a technical editor. Output STRICT JSON:
{"score":int,"issues":[{"loc":int,"issue":"...","suggestion":"..."}],"patches":[{"loc":int,"replacement":"..."}]}
USER: Draft: {draft_text}
Task: find grammar/typo/formatting issues and propose patches.


Worker should parse JSON, apply patches (or return patches for generator to use).
Acceptance: returns valid JSON parseable, score >=0.

10) Orchestrator logic — workers/tasks/generate_task.py (full flow)

Pseudo:

update_job(job_id,{state:running,progress:0.05})

retrieve_context = retriever.retrieve(prompt, top_k=4) -> join text

draft = await generator.generate(prompt, context, settings)

run technical QA -> qa_res

if qa_res.score < threshold:

build revision_prompt = combine draft + qa_res.patches/instructions

draft2 = await generator.generate(revision_prompt,...)

re-run QA (only 1 revision per phase 1 — max 2 loops)

persist version in Mongo (chapter_versions), update job result and state success

publish logs/metrics

Acceptance: job ends with state: success and result.version_id in job doc.

11) Persistence schema (Mongo) — minimal

chapter_versions:

{
  "_id": ObjectId,
  "project_id":"p1",
  "chapter_id":"c1",
  "version_number":1,
  "content":"---",
  "qa_score":85,
  "qa_issues":[...],
  "created_at":...
}


jobs as earlier.

Acceptance: stored version retrievable and content matches generated output.

12) Logs & metrics

Log job lifecycle to file/stdout and Mongo job doc.

Metrics to capture: job_count_total, job_duration_seconds, qa_scores_histogram, revision_count.

Simple: use prometheus_client exporter in worker/backend.

Acceptance: at least logs show job start/end + QA score.

13) Tests

Unit: test_chunker.py (ensure chunk sizes), test_embedder.py (embed shapes), test_technical_qa_parse.py (parse QA JSON).

Integration smoke: run scripts/ingest.py → run worker locally → POST generate job → poll job until success.

Acceptance: tests pass locally.

14) Dev run checklist (commands)

Start infra: docker compose up -d qdrant mongo redis

Install deps: pip install -r requirements.txt

Ingest sample: python scripts/ingest.py --file samples/worldbook.md --project p1

Start backend: uvicorn backend.app.main:app --reload

Start worker: python workers/worker.py (or rq worker)

Trigger: curl -X POST http://localhost:8000/generate/chapter -d @payload.json -H 'Content-Type:application/json'

Poll: GET /job/{job_id}

Acceptance Criteria (definitif untuk Phase 1)

 docker infra up (qdrant, redis, mongo)

 ingestion script ingests and Qdrant searchable

 backend offers endpoints and returns job_id on POST generate

 worker processes job and updates job state to success with version_id

 generated chapter stored in Mongo chapter_versions

 Technical QA runs and returns JSON; if score < threshold, generator revises once and final stored

 unit tests for chunker/embedder pass

 basic logs/metrics available

Notes / tradeoffs & short advices

Latency: generation+QA will be slow. Use job queue + polling UI for UX.

Costs: using OpenAI for generation costs money — for dev use local LLM or smaller model. Embeddings via E5 large self-host saves cost.

Safety: implement prompt constraints Do not invent factual claims and flag [VERIFY] in outputs.

Extensibility: Phase 2 add Structural/Character/Style QA (they require Neo4j population and more complex retrieval).