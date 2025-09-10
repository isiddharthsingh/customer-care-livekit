import json, os, re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from livekit.agents import (
    Agent, AgentSession, JobContext, RoomInputOptions, RoomOutputOptions,
    RunContext, WorkerOptions, cli, function_tool
)
from livekit.agents.voice import UserInputTranscribedEvent, ConversationItemAddedEvent
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.elevenlabs import tts as eleven_tts
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Support running as a script (python agent/main.py) and as a module (python -m agent.main)
try:
    from .graph import run_graph  # type: ignore[import-not-found]
except Exception:
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from agent.graph import run_graph  # type: ignore[no-redef]

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent / "data"
ACCOUNTS_FILE = DATA_DIR / "accounts.json"
PROFILES_FILE = DATA_DIR / "profiles.json"


def _read_json(p: Path) -> dict:
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _mask(acct: str | None) -> str:
    if not acct:
        return ""
    tail = acct[-2:]
    return f"***{tail}"


# ------------------------------------------------------------
# Account number normalization utilities
NUM_WORDS_TO_DIGIT = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

ACC_RE = re.compile(r"K\s*I\s*M\s*([0-9\s]{1,10})", re.IGNORECASE)


def _replace_number_words(text: str) -> str:
    parts = re.split(r"(\W+)", text)
    out: list[str] = []
    for tok in parts:
        low = tok.lower()
        if low.isalpha() and low in NUM_WORDS_TO_DIGIT:
            out.append(NUM_WORDS_TO_DIGIT[low])
        else:
            out.append(tok)
    return "".join(out)


def _digits_only(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def normalize_account_number(raw: str | None) -> str:
    if not raw:
        return ""
    replaced = _replace_number_words(raw.strip())
    m = ACC_RE.search(replaced)
    if not m:
        return ""
    tail = _digits_only(m.group(1))
    return f"KIM{tail}" if tail else ""


def maybe_extract_account(text: str) -> str:
    return normalize_account_number(text or "")


# script blocks
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
CJK = re.compile(r"[\u4E00-\u9FFF]")


def strong_lang_signal(text: str) -> str:
    """Return en/hi/zh based on script ratio and keywords."""
    if not text:
        return "en"
    t = text.strip()

    low = t.lower()
    if any(k in low for k in ["hindi", "हिंदी"]):
        return "hi"
    if any(k in low for k in ["chinese", "mandarin", "中文", "汉语"]):
        return "zh"
    if "english" in low:
        return "en"

    dev = len(DEVANAGARI.findall(t))
    han = len(CJK.findall(t))
    letters = sum(ch.isalpha() for ch in t)

    if letters < 6 and dev == 0 and han == 0:
        return "en"

    if han >= 2:
        return "zh"
    if letters > 0 and dev / max(letters, 1) >= 0.25:
        return "hi"

    return "en"


class KimberAssistant(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # simple in memory session store keyed by room name
        self._sessions: dict[str, dict[str, Any]] = {}

    def _sid(self, context: RunContext) -> str:
        try:
            return getattr(context.room, "name", "default") or "default"
        except Exception:
            return "default"

    def _get_session(self, context: RunContext) -> dict[str, Any]:
        sid = self._sid(context)
        return self._sessions.setdefault(sid, {"verified": False, "account_number": None, "last_intent": None})

    def _set_session(self, context: RunContext, *, verified: Optional[bool] = None, account_number: Optional[str] = None, last_intent: Optional[str] = None):
        s = self._get_session(context)
        if verified is not None:
            s["verified"] = bool(verified)
        if account_number:
            s["account_number"] = account_number
        if last_intent is not None:
            s["last_intent"] = last_intent

    # Only expose orchestrate as a tool
    @function_tool()
    async def orchestrate(
        self,
        context: RunContext,
        user_text: str,
        account_number: Optional[str] | None = None,
        proposed_address: Optional[str] | None = None,
        lang: Optional[str] | None = None,
    ) -> dict[str, Any]:
        print("[tool] orchestrate called")

        sess = self._get_session(context)

        # detect language
        if not lang:
            lang = strong_lang_signal(user_text)

        # trust current turn extraction first
        auto = maybe_extract_account(user_text)
        if auto:
            account_number = auto

        # else fall back to session account
        if not account_number and sess.get("account_number"):
            account_number = sess["account_number"]

        # honor “already verified” phrasing
        low = (user_text or "").lower()
        user_claims_verified = any(p in low for p in [
            "already verified",
            "already provided my account",
            "you verified my account",
            "you have verified my account",
            "you already verified",
        ])

        verified_flag = bool(sess.get("verified"))
        if user_claims_verified and sess.get("account_number"):
            verified_flag = True

        # run the graph with session context
        res = await run_graph(
            context,
            user_text=user_text,
            lang=lang or "en",
            tools={
                "verify_identity": self.verify_identity,
                "get_plan_info": self.get_plan_info,
                "get_profile_field": self.get_profile_field,
                "update_address": self.update_address,
                "request_transfer": self.request_transfer,
            },
            account_number=account_number,
            proposed_address=proposed_address,
            verified=verified_flag,
        )

        # update session on success
        # if verify node ran and succeeded, or any node returns verified True, keep it
        if res.get("verified") and account_number:
            norm = normalize_account_number(account_number)
            if norm:
                self._set_session(context, verified=True, account_number=norm)

        # best effort: if graph did not set verified but user just gave a clear KIM id, keep it
        if account_number and normalize_account_number(account_number):
            self._set_session(context, account_number=normalize_account_number(account_number))

        # remember last intent if present
        if "intent" in res:
            self._set_session(context, last_intent=res.get("intent"))

        reply = res.get("reply") or "Okay."
        return {"reply": reply, "lang": lang}

    # Helper methods are not tools
    async def verify_identity(self, context: RunContext, account_number: str) -> dict[str, Any]:
        norm = normalize_account_number(account_number)
        print(f"[acct] verify input='{_mask(account_number)}' -> '{_mask(norm)}'")
        if not norm:
            return {"verified": False}
        accounts = _read_json(ACCOUNTS_FILE)
        rec = accounts.get(norm)
        if not rec:
            return {"verified": False}
        return {"verified": True, "name": rec.get("name")}

    async def get_plan_info(self, context: RunContext, account_number: str) -> dict[str, Any]:
        norm = normalize_account_number(account_number)
        print(f"[acct] plan input='{_mask(account_number)}' -> '{_mask(norm)}'")
        profiles = _read_json(PROFILES_FILE)
        plan = profiles.get(norm or "", {}).get("plan", {})
        return {"plan_type": plan.get("type"), "expires": plan.get("expires")}

    async def get_profile_field(self, context: RunContext, account_number: str, field: str) -> dict[str, Any]:
        norm = normalize_account_number(account_number)
        print(f"[acct] profile input='{_mask(account_number)}' -> '{_mask(norm)}' field='{field}'")
        profiles = _read_json(PROFILES_FILE)
        return {"value": profiles.get(norm or "", {}).get(field)}

    async def update_address(self, context: RunContext, account_number: str, address: str) -> dict[str, Any]:
        norm = normalize_account_number(account_number)
        print(f"[acct] update input='{_mask(account_number)}' -> '{_mask(norm)}' address='***'")
        profiles = _read_json(PROFILES_FILE)
        key = norm or account_number
        rec = profiles.get(key) or {}
        rec["address"] = address
        profiles[key] = rec
        _write_json(PROFILES_FILE, profiles)
        return {"ok": True}

    async def request_transfer(self, context: RunContext, reason: Optional[str] = None) -> dict[str, Any]:
        text = (reason or "").lower()
        markers = ["transfer", "human", "agent", "representative", "supervisor",
                   "manager", "escalate", "talk to a person", "speak to a person"]
        return {"handoff": any(tok in text for tok in markers), "reason": reason}


async def entrypoint(ctx: JobContext):
    ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

    # simple and stable
    llm_adapter = openai.LLM(model=os.getenv("OPENAI_MODEL") or "gpt-4o-mini")

    session = AgentSession(
        stt=deepgram.STT(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            model="nova-3",
            language="multi",
            interim_results=True,
            punctuate=True,
            smart_format=True,
            endpointing_ms=300,
        ),
        llm=llm_adapter,
        tts=eleven_tts.TTS(
            api_key=ELEVEN_API_KEY,
            model="eleven_turbo_v2_5",
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        allow_interruptions=True,
        use_tts_aligned_transcript=True
    )

    async def save_transcript():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("agent_transcripts")
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{ctx.room.name}_{ts}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(session.history.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[transcript] saved {path}")

    ctx.add_shutdown_callback(save_transcript)

    @session.on("user_input_transcribed")
    def _on_user_transcribed(ev: UserInputTranscribedEvent):
        print(f"[stt][final={ev.is_final}] {ev.language}: {ev.transcript}")

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent):
        who = getattr(ev.item, "role", "unknown")
        text = getattr(ev.item, "text", None) or ""
        print(f"[history] {who}: {text}")

    await session.start(
        room=ctx.room,
        agent=KimberAssistant(instructions=(
            "You are Kimber Health. You are not UnitedHealthcare. "
            "Speak only English, Hindi, or Chinese. "
            "Default to English until the caller clearly switches. "
            "When speaking Hindi, write in Devanagari only. Avoid English words. "
            "When speaking Chinese, use simplified Chinese. "
            "Do not repeat the caller. "
            "Ask for the account number to verify. Use tools when needed. "
            "Always use the orchestrate tool to decide the next action and produce the final reply. "
            "Do not call other tools directly; orchestrate will call them for you. "
            "After verification, help with plan expiry and profile info. "
            "Update address if asked. Transfer only if the caller asks for a human. "
            "Keep replies short."
        )),
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(sync_transcription=True),
    )

    await session.generate_reply(
        instructions="Greet the caller in English and say thank you for calling Kimber Health, How may I help you today? You should always greet when you are connected to the caller. Then when the user greets or asks anything, ask for their account number for verification. Keep it short."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))