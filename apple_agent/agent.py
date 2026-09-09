from __future__ import annotations

from dataclasses import asdict, dataclass

from apple_agent.draft import draft_reply
from apple_agent.escalate import RouteDecision, decide
from apple_agent.intents import IntentClassifier, IntentPrediction
from apple_agent.retrieve import TfidfRetriever


@dataclass
class AgentResult:
    intent: str
    intent_confidence: float
    weak_intent: str
    route: str
    reason: str
    reply: str
    retrieval_score: float
    retrieved_ids: list[str]
    signals: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


class SupportAgent:
    def __init__(self, classifier: IntentClassifier, retriever: TfidfRetriever):
        self.classifier = classifier
        self.retriever = retriever

    def handle(self, text: str, k: int = 8) -> AgentResult:
        pred: IntentPrediction = self.classifier.predict(text)
        neighbors = self.retriever.query(text, k=k, intent=pred.intent)
        top_score = neighbors[0]["score"] if neighbors else 0.0
        route: RouteDecision = decide(text, pred, top_score)
        reply = draft_reply(text, pred, route, neighbors)
        return AgentResult(
            intent=pred.intent,
            intent_confidence=round(pred.confidence, 4),
            weak_intent=pred.weak_intent,
            route=route.route,
            reason=route.reason,
            reply=reply,
            retrieval_score=round(float(top_score), 4),
            retrieved_ids=[n["id"] for n in neighbors],
            signals=route.signals,
        )
