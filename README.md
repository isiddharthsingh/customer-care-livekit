# Kimber Health Voice Agent with Turn Detection (LiveKit + Deepgram)

This starter gives you:
- LiveKit Agents with Deepgram STT/TTS
- Silero VAD and LiveKit Turn Detector for strong end-of-turn
- JSON data store (no DB yet)
- Transcripts saved to disk
- Room recordings via LiveKit Egress (S3 recommended)

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
