from __future__ import annotations

import re
from typing import TypedDict, Optional, Callable, Awaitable, Any

from langgraph.graph import StateGraph, END

ToolFn = Callable[..., Awaitable[dict[str, Any]]]


class OrchestratorState(TypedDict, total=False):
    context: Any
    user_text: str
    account_number: Optional[str]
    proposed_address: Optional[str]
    verified: bool
    intent: str
    target_intent: str
    route: str
    reply: str
    lang: str


def _classify_intent(text: str) -> str:
    t = (text or "").lower()

    def has_word(pat: str) -> bool:
        return re.search(rf"\b{pat}\b", t) is not None

    # user asks for a person
    if any(k in t for k in ["human", "transfer", "agent", "representative", "supervisor", "escalate"]):
        return "transfer"

    # user says they are already verified
    if any(p in t for p in [
        "already verified",
        "you verified my account",
        "you have verified my account",
        "already provided my account",
        "you already verified",
    ]):
        return "ack_verified"

    # read address (do not confuse with update)
    if ("address" in t and any(q in t for q in ["what is", "what's", "tell", "show", "on file", "my address"])) and not any(
        k in t for k in ["change", "update", "new address", "update my address", "change address"]
    ):
        return "get_address"

    # change address
    if ("address" in t and "email" not in t) or "update address" in t or "change address" in t or has_word("address"):
        return "update_address"

    # plan info
    if any(k in t for k in ["plan", "expiry", "expires", "coverage", "plan details"]):
        return "plan_info"

    # avoid triggering on plain "kim"
    if any(k in t for k in ["account", "verify", "verification"]):
        return "verify"

    return "smalltalk"


# cache compiled graphs per tool set
_CACHE: dict[tuple[tuple[str, int], ...], Any] = {}


def build_graph(tools: dict[str, ToolFn]):
    key = tuple(sorted((name, id(fn)) for name, fn in tools.items()))
    if key in _CACHE:
        return _CACHE[key]

    g = StateGraph(OrchestratorState)

    def classify(state: OrchestratorState):
        intent = _classify_intent(state.get("user_text") or "")
        return {"intent": intent, "target_intent": intent}

    def decide(state: OrchestratorState):
        intent = state.get("intent")
        verified = bool(state.get("verified"))
        if intent in ("plan_info", "update_address", "get_address") and not verified:
            return {"route": "verify"}
        return {"route": intent}

    async def verify(state: OrchestratorState):
        acct = state.get("account_number")
        if not acct:
            return {"reply": "Please share your account number to verify."}
        r = await tools["verify_identity"](state["context"], account_number=acct)  # type: ignore[index]
        ok = bool(r.get("verified"))
        if not ok:
            return {"verified": False, "reply": "I could not verify that account. Please say the number again."}
        target = state.get("target_intent") or "smalltalk"
        # keep verified in state so the caller can cache it
        return {"verified": True, "intent": target, "reply": "Verified."}

    async def plan_info(state: OrchestratorState):
        if not state.get("verified"):
            return {"reply": "I can help with that. First, your account number please."}
        r = await tools["get_plan_info"](state["context"], account_number=state.get("account_number"))  # type: ignore[index]
        t, exp = r.get("plan_type"), r.get("expires")
        return {"reply": f"Your plan is {t or 'unknown'} and expires on {exp or 'unknown'}.", "verified": True}

    async def update_address(state: OrchestratorState):
        if not state.get("verified"):
            return {"reply": "I can update that. First, your account number please."}
        addr = state.get("proposed_address")
        if not addr:
            return {"reply": "What is the new address?", "verified": True}
        await tools["update_address"](state["context"], account_number=state.get("account_number"), address=addr)  # type: ignore[index]
        return {"reply": f"I updated the address to: {addr}.", "verified": True}

    async def get_address(state: OrchestratorState):
        if not state.get("verified"):
            return {"reply": "I can help with that. First, your account number please."}
        r = await tools["get_profile_field"](state["context"], account_number=state.get("account_number"), field="address")  # type: ignore[index]
        val = r.get("value")
        if val:
            return {"reply": f"Your address on file is: {val}.", "verified": True}
        return {"reply": "I do not see an address on file.", "verified": True}

    async def transfer(state: OrchestratorState):
        r = await tools["request_transfer"](state["context"], reason=state.get("user_text"))  # type: ignore[index]
        return {"reply": ("Okay, I will transfer you to a human representative." if r.get("handoff") else "I can help here. What do you need?")}

    def ack_verified(state: OrchestratorState):
        # trust prior verification if caller asserts and upstream passed verified/account
        if state.get("account_number"):
            return {"verified": True, "reply": "You are verified. What would you like to do next?"}
        return {"reply": "Please share your account number to verify."}

    def smalltalk(state: OrchestratorState):
        return {"reply": "How can I help you today?"}

    g.add_node("classify", classify)
    g.add_node("decide", decide)
    g.add_node("verify", verify)
    g.add_node("plan_info", plan_info)
    g.add_node("update_address", update_address)
    g.add_node("get_address", get_address)
    g.add_node("transfer", transfer)
    g.add_node("ack_verified", ack_verified)
    g.add_node("smalltalk", smalltalk)

    g.set_entry_point("classify")
    g.add_edge("classify", "decide")
    g.add_conditional_edges("decide", lambda s: s.get("route"), {
        "verify": "verify",
        "plan_info": "plan_info",
        "update_address": "update_address",
        "get_address": "get_address",
        "transfer": "transfer",
        "ack_verified": "ack_verified",
        "smalltalk": "smalltalk",
    })

    def _post_verify_route(state: OrchestratorState):
        if not state.get("verified"):
            return "END"
        tgt = state.get("intent")
        if tgt in ("plan_info", "update_address", "get_address"):
            return tgt
        return "END"

    g.add_conditional_edges("verify", _post_verify_route, {
        "plan_info": "plan_info",
        "update_address": "update_address",
        "get_address": "get_address",
        "END": END,
    })

    for n in ["plan_info", "update_address", "get_address", "transfer", "ack_verified", "smalltalk"]:
        g.add_edge(n, END)

    compiled = g.compile()
    _CACHE[key] = compiled
    return compiled


async def run_graph(context: Any, user_text: str, tools: dict[str, ToolFn], lang: str = "en", **slots: Any) -> dict[str, Any]:
    graph = build_graph(tools)
    state: OrchestratorState = {
        "context": context,
        "user_text": user_text,
        "lang": lang,
        **slots,
    }
    out = await graph.ainvoke(state)
    # always include reply key for caller convenience
    if "reply" not in out:
        out["reply"] = "Okay."
    return out