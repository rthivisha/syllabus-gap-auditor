# Syllabus Gap Audit AI — Project & System Architecture Documentation

## 1. Project Overview & Purpose
**Syllabus Gap Audit AI** compares a YouTube lecture's spoken transcript and slide text against a selected reference PDF, such as a textbook, syllabus, or manual. It checks which document topics the video covers and identifies verified omissions, including skipped mechanisms. The report focuses on missed topics and grounds them in reference sections, PDF pages, and supporting textbook quotes; video windows are checked for coverage evidence.

## 2. How to Get Started (Step-by-Step User Workflow)
1. **Select or Upload a Reference PDF:** Drop a PDF textbook or syllabus into the upload bar on the main screen (which parses, chunks, and indexes it into Pinecone before the upload completes, then activates it) or click one of the ready sample books.
2. **Choose Audit Depth Mode:** In the floating AI Composer pill, toggle between **⚡ Fast** (rapid section-level scan) or **🔭 Deep Research** (strict mechanism-level rubric evaluation with automated repair retries).
3. **Apply a YouTube Lecture URL:** Paste the YouTube video link you want to audit and specify whether to analyze the **entire book** or **specific chapters/sections**.
4. **Ask About the PDF:** Ask a textbook question or choose a suggested question to retrieve relevant passages and receive an answer with page and section citations.
5. **Inspect Results, Citations & HITL Queue:** Review the structured gap report, expand the **Agent Execution Trace** and **Retrieved Reference Evidence** chunks, or verify borderline topics routed to the **✅ Approvals Queue** and **📊 Live Metrics** tabs.

## 3. Complete Tech Stack
- **Frontend Interface:** Single-page Streamlit application featuring a floating white-pill **AI Composer** with cycling blur-rise prompts, a Python-side UI bridge (eliminating browser iframe CORS/PNA blocks), non-blocking `@st.fragment` live stopwatch polling, an Approvals Queue, and a Live Metrics dashboard.
- **Backend & Stateful Agent Orchestration:** **FastAPI** + **LangGraph (`StateGraph`)** with a `MemorySaver` checkpointer and `interrupt()` conversational loop. Intent routing separates general/onboarding questions (`project_guide`), Pinecone-grounded textbook answers (`pdf_qa`), and YouTube audits (`media_ingestion -> scoped_retriever -> gap_auditor -> evaluator -> repair -> finalizer`).
- **LLM Gateway & Automated Failover:** Server-configured Gemini and Groq use **Cloudflare AI Gateway** with automatic Gemini-to-Groq failover. Optional per-request Gemini/Groq BYOK credentials use direct provider connections, stay out of graph checkpoints, and are not persisted.
- **Multimodal Video Ingestion:** Ultra-fast YouTube caption extraction via `youtube-transcript-api` and `yt-dlp --write-auto-sub` (`json3`) (~1.5s), with **Groq Whisper (`whisper-large-v3-turbo`)** fallback, segmented into 25-second timestamped windows.
- **Hybrid Retrieval & Cross-Encoder Reranking:** Page-aware child chunks are coalesced up to 1,500 characters, then indexed with section-summary vectors in **Pinecone** using **`BAAI/bge-base-en-v1.5`** dense embeddings, paired with **FlashRank (`ms-marco-MiniLM-L-12-v2`)** cross-encoder reranking.
- **Persistence, Caching & Governance:** **Neon PostgreSQL** for persistent chat sessions, audit execution logs, a unified five-turn guest LLM quota (BYOK turns are unlimited), and Human-in-the-Loop (HITL) escalations; local SQLite and JSON caching for video transcripts and precomputed BGE window embeddings.

## 4. Key Efficiency & Accuracy Engineering Techniques
- **Persistent BGE Window Embedding Cache:** 25-second video window embeddings are cached to disk on first ingestion, reducing repeat BGE embedding latency from ~14 seconds to `0.00s`.
- **Single-Query Scoped Pinecone + Video-vs-Section FlashRank:** Fetches all scoped textbook child/parent chunks in a single batched Pinecone metadata query and runs FlashRank directly between each textbook section and its top-3 candidate video windows (`< 1.5s` total).
- **Anti-Lenience Granularity Rubric (`skills.md`):** Enforces mechanism-level verification so introductory lectures are not excused when concrete rules, code invariants, or syllabus subtopics from the textbook are omitted.
- **Intent-Gated Quota Protection:** Greetings and deterministic video-scope prompts do not use quota. Successful project-guide, PDF-Q&A, and audit model turns share a five-turn guest limit; requests with a BYOK key bypass that limit.
