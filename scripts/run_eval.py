"""Reproduce headline numbers on the golden set (< 15 min after install)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apple_agent.agent import SupportAgent
from apple_agent.baselines import copy_nearest, trivial_handle
from apple_agent.config import GOLDEN_PATH, HUMAN_JUDGE_PATH, INDEX_DIR, RESULTS_DIR
from apple_agent.evaluate import (
    confusion,
    intent_metrics,
    judge_batch,
    load_jsonl,
    pick_failures,
    reply_overlap_metrics,
    route_metrics,
    safety_flags,
)
from apple_agent.intents import IntentClassifier
from apple_agent.judge import JudgeScore, agreement
from apple_agent.retrieve import TfidfRetriever


def run_system(name: str, gold: list[dict], handle, judge_backend: str = "programmatic") -> dict:
    results = [handle(g["text"]) for g in gold]
    y_true_i = [g["intent"] for g in gold]
    y_pred_i = [r.intent for r in results]
    y_true_r = [g["route"] for g in gold]
    y_pred_r = [r.route for r in results]
    judged = judge_batch(gold, results, backend=judge_backend)
    out = {
        "n": len(gold),
        "intent": intent_metrics(y_true_i, y_pred_i),
        "route": route_metrics(y_true_r, y_pred_r),
        "reply_vs_historical_apple": reply_overlap_metrics(
            [r.reply for r in results], [g.get("apple_reply", "") for g in gold]
        ),
        "safety": safety_flags(results),
        "judge": {
            "mean": judged["mean"],
            "pass_at_7_rate": judged["pass_at_7_rate"],
            "pass_at_9_rate": judged["pass_at_9_rate"],
        },
        "intent_confusion": confusion(y_true_i, y_pred_i),
        "failures": pick_failures(gold, results),
    }
    # persist per-example predictions for the full agent only
    if name == "agent":
        pred_path = RESULTS_DIR / "golden_predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for g, r, js in zip(gold, results, judged["scores"]):
                f.write(
                    json.dumps(
                        {
                            "id": g["id"],
                            "text": g["text"],
                            "gold_intent": g["intent"],
                            "pred_intent": r.intent,
                            "gold_route": g["route"],
                            "pred_route": r.route,
                            "reason": r.reason,
                            "reply": r.reply,
                            "judge": js,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        out["predictions_path"] = str(pred_path)
    _ = name
    return out


def human_agreement() -> dict | None:
    if not HUMAN_JUDGE_PATH.exists():
        return None
    rows = load_jsonl(HUMAN_JUDGE_PATH)
    human = [
        JudgeScore(
            groundedness=r["groundedness"],
            helpfulness=r["helpfulness"],
            brand_voice=r["brand_voice"],
            safety=r["safety"],
            routing_consistency=r["routing_consistency"],
            rationale=r.get("rationale", "human"),
            backend="human",
        )
        for r in rows
    ]
    auto = [
        JudgeScore(
            groundedness=r["auto_groundedness"],
            helpfulness=r["auto_helpfulness"],
            brand_voice=r["auto_brand_voice"],
            safety=r["auto_safety"],
            routing_consistency=r["auto_routing_consistency"],
            rationale="programmatic",
            backend="programmatic",
        )
        for r in rows
    ]
    return agreement(human, auto)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["programmatic", "openai"], default="programmatic")
    args = parser.parse_args()
    if not (INDEX_DIR / "intent.joblib").exists():
        raise SystemExit("Index missing. Run: python scripts/build_index.py")
    if args.judge == "openai":
        print("Scoring with OpenAI judge (agent drafts only). Baselines stay programmatic.")
    gold = load_jsonl(GOLDEN_PATH)
    clf = IntentClassifier.load(INDEX_DIR / "intent.joblib")
    retr = TfidfRetriever.load(INDEX_DIR)
    agent = SupportAgent(clf, retr)

    systems = {
        "trivial": lambda t: trivial_handle(t),
        "copy_nearest": lambda t: copy_nearest(t, retr),
        "agent": lambda t: agent.handle(t),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"n_golden": len(gold), "systems": {}}
    for name, fn in systems.items():
        print("evaluating", name)
        backend = args.judge if name == "agent" else "programmatic"
        report["systems"][name] = run_system(name, gold, fn, judge_backend=backend)
        s = report["systems"][name]
        print(
            f"  intent_acc={s['intent']['accuracy']} macroF1={s['intent']['macro_f1']} "
            f"route_acc={s['route']['accuracy']} false_auto={s['route']['false_auto_count']} "
            f"judge_total={s['judge']['mean']['total']} pass@7={s['judge']['pass_at_7_rate']}"
        )

    report["judge_human_agreement"] = human_agreement()
    out = RESULTS_DIR / "metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote", out)
    _headline(report)


def _headline(report: dict) -> None:
    a = report["systems"]["agent"]
    print("\n=== HEADLINE ===")
    print(f"Golden n={report['n_golden']}")
    print(f"Intent accuracy {a['intent']['accuracy']} / macro-F1 {a['intent']['macro_f1']}")
    print(
        f"Route accuracy {a['route']['accuracy']} / escalate recall {a['route']['escalate_recall']} "
        f"/ false-auto {a['route']['false_auto_count']}"
    )
    print(
        f"Judge mean {a['judge']['mean']['total']}/10 / pass@7 {a['judge']['pass_at_7_rate']} "
        f"/ pass@9 {a['judge'].get('pass_at_9_rate')}"
    )
    if report.get("judge_human_agreement"):
        print("Judge-human agreement:", report["judge_human_agreement"])


if __name__ == "__main__":
    main()
