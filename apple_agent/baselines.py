from __future__ import annotations

from apple_agent.agent import AgentResult
from apple_agent.retrieve import TfidfRetriever
from apple_agent import textutil as T


CANNED = (
    "We're sorry you're having trouble. Please contact Apple Support and we'll be happy to help."
)


def trivial_handle(text: str) -> AgentResult:
    _ = text
    return AgentResult(
        intent="other",
        intent_confidence=1.0,
        weak_intent="other",
        route="escalate",
        reason="Trivial baseline always escalates with a canned apology.",
        reply=CANNED,
        retrieval_score=0.0,
        retrieved_ids=[],
        signals=["trivial"],
    )


def copy_nearest(text: str, retriever: TfidfRetriever) -> AgentResult:
    """Simple baseline: copy the historically nearest Apple reply."""
    hits = retriever.query(text, k=1)
    if not hits:
        return trivial_handle(text)
    nb = hits[0]
    reply = T.MENTION.sub("@customer", nb.get("apple_reply") or CANNED)
    asked_dm = "dm" in reply.lower()
    return AgentResult(
        intent=nb.get("weak_intent") or "other",
        intent_confidence=float(nb.get("score") or 0.0),
        weak_intent=nb.get("weak_intent") or "other",
        route="escalate" if asked_dm else "auto",
        reason=(
            "Copied nearest historical Apple reply; routed escalate iff that reply asked for a DM."
        ),
        reply=reply,
        retrieval_score=float(nb.get("score") or 0.0),
        retrieved_ids=[nb["id"]],
        signals=["copy_nearest"],
    )
