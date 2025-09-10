from __future__ import annotations

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
    reply: str
    lang: str


def _classify_intent(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["human", "transfer", "agent", "representative", "supervisor", "escalate"]):
        return "transfer"
    if any(k in t for k in ["address", "change address", "update address"]):
        return "update_address"
    if any(k in t for k in ["plan", "expiry", "expires", "coverage"]):
        return "plan_info"
    if any(k in t for k in ["account", "kim", "verify", "verification"]):
        return "verify"
    return "smalltalk"


def build_graph(tools: dict[str, ToolFn]):
    g = StateGraph(OrchestratorState)

    def classify(state: OrchestratorState):
        return {"intent": _classify_intent(state.get("user_text") or "")}

    async def verify(state: OrchestratorState):
        if not state.get("account_number"):
            return {"reply": "Please share your account number so I can verify."}
        r = await tools["verify_identity"](state["context"], account_number=state["account_number"])  # type: ignore[index]
        ok = bool(r.get("verified"))
        return {"verified": ok, "reply": ("Verified." if ok else "I couldn’t verify that account. Please repeat the number.")}

    async def plan_info(state: OrchestratorState):
        if not state.get("verified"):
            return {"reply": "I’ll help with that. First, please share your account number to verify."}
        r = await tools["get_plan_info"](state["context"], account_number=state.get("account_number"))  # type: ignore[index]
        t, exp = r.get("plan_type"), r.get("expires")
        return {"reply": f"Your plan is {t or 'unknown'} and expires on {exp or 'unknown'}."}

    async def update_address(state: OrchestratorState):
        if not state.get("verified"):
            return {"reply": "I can update that. First, please share your account number to verify."}
        addr = state.get("proposed_address")
        if not addr:
            return {"reply": "What is the new address?"}
        await tools["update_address"](state["context"], account_number=state.get("account_number"), address=addr)  # type: ignore[index]
        return {"reply": "Your address is updated."}

    async def transfer(state: OrchestratorState):
        r = await tools["request_transfer"](state["context"], reason=state.get("user_text"))  # type: ignore[index]
        return {"reply": ("Okay, I’ll transfer you to a human representative." if r.get("handoff") else "I can help here. What do you need?")}

    def smalltalk(state: OrchestratorState):
        return {"reply": "How can I help you today?"}

    g.add_node("classify", classify)
    g.add_node("verify", verify)
    g.add_node("plan_info", plan_info)
    g.add_node("update_address", update_address)
    g.add_node("transfer", transfer)
    g.add_node("smalltalk", smalltalk)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", lambda s: s.get("intent"), {
        "verify": "verify",
        "plan_info": "plan_info",
        "update_address": "update_address",
        "transfer": "transfer",
        "smalltalk": "smalltalk",
    })
    for n in ["verify", "plan_info", "update_address", "transfer", "smalltalk"]:
        g.add_edge(n, END)
    return g.compile()


async def run_graph(context: Any, user_text: str, tools: dict[str, ToolFn], lang: str = "en", **slots: Any) -> str:
    graph = build_graph(tools)
    state: OrchestratorState = {
        "context": context,
        "user_text": user_text,
        "lang": lang,
        **slots,
    }
    out = await graph.ainvoke(state)
    return out.get("reply") or "Okay."


