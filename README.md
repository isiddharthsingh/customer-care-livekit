# Kimber Health Voice Agent with Turn Detection (LiveKit + Deepgram)

This starter gives you:
- LiveKit Agents with Deepgram STT/TTS
- Silero VAD and LiveKit Turn Detector for strong end-of-turn
- JSON data store (no DB yet)
- Transcripts saved to disk
- Room recordings via LiveKit Egress (S3 recommended)

## LangGraph integration

This project includes a LangGraph-based orchestrator and uses the LiveKit LangChain/LangGraph adapter as the default LLM.

- Dependencies are in `agent/requirements.txt`:
  - `langgraph`, `langchain-openai`, `livekit-plugins-langchain`
- The orchestrator lives in `agent/graph.py` and is exposed as an Agent tool `orchestrate` inside `KimberAssistant`.
- Agent instructions prefer using `orchestrate` to route intents and call tools (`verify_identity`, `get_plan_info`, `update_address`, `request_transfer`).

The LiveKit LangChain adapter runs a LangChain Runnable (default: `ChatOpenAI`). You can replace it with a compiled LangGraph workflow to make the graph act as the LLM.

References:
- LiveKit x LangChain integration: [docs.livekit.io](https://docs.livekit.io/agents/integrations/llm/langchain/)
- LangGraph setup: [langchain-ai.github.io](https://langchain-ai.github.io/langgraph/cloud/deployment/setup/)

### Why LangGraph here
- Deterministic control and guardrails (verify before plan/address; confirm updates; explicit transfer only)
- Centralized tool sequencing via `orchestrate` for consistent, testable flows
- Easy to add skills/nodes without changing STT/TTS/turn logic
- Lower hallucinations by constraining actions and when tools may run
- Optional LiveKit LangChain adapter path to run any LangChain Runnable (incl. compiled LangGraph)

### LiveKit LangGraph plugin is default
No extra configuration is needed. To run:

```bash
python agent/main.py dev
```

To change the OpenAI model used by the adapter, set `OPENAI_MODEL` in your `.env`.

### New use cases you can add quickly
- KYC/verification gates before sensitive actions
- Benefit/plan RAG: retrieval node for PDFs/KBs when needed
- Address update confirmations with audit fields
- Claims/billing intents with backend API calls
- Policy-driven human handoff and reason logging
- Post-call CRM notes/tickets and call summaries
- A/B flows by routing subsets of calls to alternative scripts
- Optional checkpointing/memory to persist slots across interruptions

## Quick start

1) Copy `.env.example` to `.env` and fill values.
2) Create a venv and install:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r agent/requirements.txt
pip install -r server/requirements.txt
```
3) Start the token and recording server:
```bash
uvicorn server.main:app --reload --port 8787
```
4) Start the agent worker:
```bash
python agent/main.py dev
```
5) Serve the web client:
```bash
python -m http.server 5500 -d web
```
Open http://localhost:5500 and click Join.

Notes:
- Turn detector and Silero may download model files on first run.
- Recordings: set S3 env vars to save MP4/HLS in your bucket.
- Replace JSON with a real DB later. Keep tool signatures the same.
