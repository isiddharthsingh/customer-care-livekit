# agent/main.py
import json, os, time, re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from collections import deque

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
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

class KimberAssistant(Agent):
    @function_tool()
    async def orchestrate(self, context: RunContext, user_text: str, account_number: Optional[str] | None = None, proposed_address: Optional[str] | None = None, lang: Optional[str] | None = None) -> dict[str, Any]:
        reply = await run_graph(
            context,
            user_text=user_text,
            lang=lang or "en",
            tools={
                "verify_identity": self.verify_identity,
                "get_plan_info": self.get_plan_info,
                "update_address": self.update_address,
                "request_transfer": self.request_transfer,
            },
            account_number=account_number,
            proposed_address=proposed_address,
        )
        return {"reply": reply}
    @function_tool()
    async def verify_identity(self, context: RunContext, account_number: str) -> dict[str, Any]:
        norm = normalize_account_number(account_number)
        print(f"[acct] verify input='{account_number}' -> '{norm}'")
        if not norm:
            return {"verified": False}
        accounts = _read_json(ACCOUNTS_FILE)
        rec = accounts.get(norm)
        if not rec:
            return {"verified": False}
        return {"verified": True, "name": rec.get("name")}

    @function_tool()
    async def get_plan_info(self, context: RunContext, account_number: str) -> dict[str, Any]:
        norm = normalize_account_number(account_number)
        print(f"[acct] plan input='{account_number}' -> '{norm}'")
        profiles = _read_json(PROFILES_FILE)
        plan = profiles.get(norm or "", {}).get("plan", {})
        return {"plan_type": plan.get("type"), "expires": plan.get("expires")}

    @function_tool()
    async def get_profile_field(self, context: RunContext, account_number: str, field: str) -> dict[str, Any]:
        norm = normalize_account_number(account_number)
        print(f"[acct] profile input='{account_number}' -> '{norm}' field='{field}'")
        profiles = _read_json(PROFILES_FILE)
        return {"value": profiles.get(norm or "", {}).get(field)}

    @function_tool()
    async def update_address(self, context: RunContext, account_number: str, address: str) -> dict[str, Any]:
        norm = normalize_account_number(account_number)
        print(f"[acct] update input='{account_number}' -> '{norm}' address='{address}'")
        profiles = _read_json(PROFILES_FILE)
        key = norm or account_number
        rec = profiles.get(key) or {}
        rec["address"] = address
        profiles[key] = rec
        _write_json(PROFILES_FILE, profiles)
        return {"ok": True}

    @function_tool()
    async def request_transfer(self, context: RunContext, reason: Optional[str] = None) -> dict[str, Any]:
        text = (reason or "").lower()
        markers = ["transfer", "human", "agent", "representative", "supervisor",
                   "manager", "escalate", "talk to a person", "speak to a person"]
        return {"handoff": any(tok in text for tok in markers), "reason": reason}

# ------------------------------------------------------------
# Account number normalization utilities
# Accept inputs like:
#   "K I m 4 0 0 4", "kim4004", "kIm-4004.", "My account is KIM4004"
# Output canonical form: "KIM4004"
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

def _replace_number_words(text: str) -> str:
    # Convert standalone number words to digits (e.g., "four zero zero four" -> "4004")
    # Keep separators so we don't glue unrelated words together
    parts = re.split(r"(\W+)", text)
    out: list[str] = []
    for tok in parts:
        low = tok.lower()
        if low.isalpha() and low in NUM_WORDS_TO_DIGIT:
            out.append(NUM_WORDS_TO_DIGIT[low])
        else:
            out.append(tok)
    return "".join(out)

def normalize_account_number(raw: str | None) -> str:
    if not raw:
        return ""
    # First, map spelled-out digit words to digits before stripping
    replaced = _replace_number_words(raw.strip())
    s = replaced.upper()
    # keep only letters and digits
    s = "".join(ch for ch in s if ch.isalnum())
    # collapse repeated KIM and digits
    # find suffix 1-6 digits at end
    digits = []
    for ch in reversed(s):
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if digits:
        suffix = "".join(reversed(digits))
        return f"KIM{suffix}"
    # if already like KIM####
    if s.startswith("KIM") and any(ch.isdigit() for ch in s[3:]):
        tail = "".join(ch for ch in s[3:] if ch.isdigit())
        return f"KIM{tail}"
    return s

# script blocks
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
CJK = re.compile(r"[\u4E00-\u9FFF]")

def strong_lang_signal(text: str) -> str:
    """Return en/hi/zh based on script ratio and keywords."""
    if not text:
        return "en"
    t = text.strip()

    # explicit asks
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

    # avoid switching on short junk like "Media", "Bolo."
    if letters < 6 and dev == 0 and han == 0:
        return "en"

    if han >= 2:
        return "zh"
    if letters > 0 and dev / max(letters, 1) >= 0.25:
        return "hi"

    return "en"

async def entrypoint(ctx: JobContext):
    ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

    VOICE_MAP = {
        "en": os.getenv("ELEVEN_VOICE_EN"),
        "hi": os.getenv("ELEVEN_VOICE_HI"),
        "zh": os.getenv("ELEVEN_VOICE_ZH"),
    }
    ALLOWED = {"en", "hi", "zh"}

    def norm_lang(code: str) -> str:
        if not code:
            return "en"
        b = code.split("-")[0].lower()
        if b.startswith("cmn"):
            b = "zh"
        return b if b in ALLOWED else "en"

    # Default: use LiveKit's LangChain/LangGraph adapter backed by a LangChain Runnable.
    # Swap ChatOpenAI for a compiled LangGraph workflow when ready.
    try:
        from livekit.plugins import langchain as lk_langchain  # type: ignore
        from langchain_openai import ChatOpenAI  # type: ignore
        openai_model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        chat_runnable = ChatOpenAI(model=openai_model, temperature=0.2)
        llm_adapter = lk_langchain.LLMAdapter(runnable=chat_runnable)
    except Exception as e:
        print(f"[langgraph] Adapter unavailable, falling back to OpenAI LLM: {e}")
        llm_adapter = openai.LLM(model="gpt-4o-mini")

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
            language="en",
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        allow_interruptions=True,
        use_tts_aligned_transcript=True
    )

    # save transcript on shutdown
    async def save_transcript():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("agent_transcripts"); out.mkdir(parents=True, exist_ok=True)
        path = out / f"{ctx.room.name}_{ts}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(session.history.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[transcript] saved {path}")
    ctx.add_shutdown_callback(save_transcript)

    # sticky switching
    current_lang = "en"
    last_switch = 0.0
    SWITCH_COOLDOWN = 2.0
    last_signals: deque[str] = deque(maxlen=2)

    def maybe_switch(target: str, reason: str):
        nonlocal current_lang, last_switch
        target = norm_lang(target)
        if target == current_lang:
            return
        now = time.time()
        if now - last_switch < SWITCH_COOLDOWN:
            return
        # Switch language on the existing TTS engine (SDK 1.2.x has no set_tts)
        session.tts.update_options(language=target)
        current_lang = target
        last_switch = now
        print(f"[tts] switched -> {current_lang} ({reason}) voice_id={VOICE_MAP.get(current_lang) or 'default'}")

    @session.on("user_input_transcribed")
    def _on_user_transcribed(ev: UserInputTranscribedEvent):
        print(f"[stt][final={ev.is_final}] {ev.language}: {ev.transcript}")
        if not ev.is_final:
            return

        # compute a signal from content, not only STT tag
        sig = strong_lang_signal(ev.transcript or "")
        last_signals.append(sig)

        # require two matching signals to switch, unless the user asked explicitly
        explicit = sig in {"hi", "zh"} and sig != current_lang and (
            ("hindi" in (ev.transcript or "").lower()) or
            any(k in (ev.transcript or "") for k in ["हिंदी", "中文", "汉语", "Mandarin", "Chinese"])
        )

        if explicit:
            maybe_switch(sig, "explicit")
            return

        if len(last_signals) == 2 and last_signals[0] == last_signals[1] and last_signals[0] != current_lang:
            maybe_switch(last_signals[0], "stable-two-signals")
            return

        # As a fallback, honor STT language if very confident script wise
        if ev.language and norm_lang(ev.language) != current_lang:
            if sig in {"hi", "zh"}:
                maybe_switch(sig, "stt+script")

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

    # force first greet to English
    await session.generate_reply(
        instructions="Greet the caller in English and say thank you for calling Kimber Health, How may I help you today? you should alsways greet when you are connected to the caller. Then when the user also greet or asks anything, ask them their account number for verification. Keep it short."
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
