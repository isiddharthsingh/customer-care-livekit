# server/main.py
import os, time
from typing import Optional
from pathlib import Path
import aiohttp
from datetime import timedelta
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from livekit import api

load_dotenv()
app = FastAPI()

# CORS: allow local web client (127.0.0.1 and localhost on common dev ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

class TokenReq(BaseModel):
    room: str
    identity: str

class TokenResp(BaseModel):
    access_token: str
    url: str

@app.post("/token", response_model=TokenResp)
def create_token(req: TokenReq):
    grants = api.VideoGrants(
        room=req.room,
        room_join=True,
        can_publish=True,
        can_subscribe=True,
        room_record=True,
    )
    at = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(req.identity)
        .with_grants(grants)
        .with_ttl(timedelta(seconds=3600))
    )
    return TokenResp(access_token=at.to_jwt(), url=LIVEKIT_URL)

# Egress is handled via livekit.api.egress_service.EgressService (async)

class StartRecordReq(BaseModel):
    room: str
    audio_only: bool = True
    path_prefix: Optional[str] = "recording"

class StartRecordResp(BaseModel):
    egress_id: str

@app.post("/record/start", response_model=StartRecordResp)
async def start_recording(req: StartRecordReq):
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / (req.path_prefix or "recording")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    filename = f"{req.room}_{ts}.mp4"

    s3_key = os.getenv("S3_ACCESS_KEY")
    s3_secret = os.getenv("S3_SECRET")
    s3_bucket = os.getenv("S3_BUCKET")
    s3_region = os.getenv("S3_REGION") or ""
    s3_endpoint = os.getenv("S3_ENDPOINT") or ""

    if s3_key and s3_secret and s3_bucket:
        key_prefix = (req.path_prefix or "recording").strip("/")
        file = api.EncodedFileOutput(
            s3=api.S3Upload(
                access_key=s3_key,
                secret=s3_secret,
                bucket=s3_bucket,
                region=s3_region,
                endpoint=s3_endpoint,
            ),
            filepath=f"{key_prefix}/{filename}",
            file_type=api.EncodedFileType.MP4,
        )
    else:
        # fallback: local path (works only on self-hosted egress)
        file = api.EncodedFileOutput(
            filepath=str(out_dir / filename),
            file_type=api.EncodedFileType.MP4,
        )

    async with aiohttp.ClientSession() as session:
        svc = api.egress_service.EgressService(session, LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        info: api.EgressInfo = await svc.start_room_composite_egress(
            api.RoomCompositeEgressRequest(room_name=req.room, audio_only=req.audio_only, file=file)
        )
    return StartRecordResp(egress_id=info.egress_id)

class StopRecordReq(BaseModel):
    egress_id: str

@app.post("/record/stop")
async def stop_recording(req: StopRecordReq):
    async with aiohttp.ClientSession() as session:
        svc = api.egress_service.EgressService(session, LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        await svc.stop_egress(api.StopEgressRequest(egress_id=req.egress_id))
    return {"stopped": True}
