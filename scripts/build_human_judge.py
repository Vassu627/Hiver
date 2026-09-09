"""Human rubric scores on 40 agent drafts (same 0-2 x 5 dimensions as the judge)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apple_agent.config import HUMAN_JUDGE_PATH, RESULTS_DIR

# id -> (groundedness, helpfulness, brand_voice, safety, routing_consistency, note)
# Scored against the LLM-as-judge rubric, independently of the programmatic backend.
HUMAN = {
    "6af36d260e7b": (1, 1, 2, 2, 0, "auto-sent a generic About-path; does not say how to remove duplicate ringtones"),
    "129d8c534c5a": (1, 1, 2, 2, 0, "screenshot-dependent how-to; draft never looks at 'this way'"),
    "9a72bb5fb2a7": (1, 1, 2, 2, 1, "customer wants to wipe game data, not recover a photo library"),
    "0732aa3b0b3d": (0, 1, 2, 2, 1, "invents battery drain; they asked how to show battery %"),
    "e4a2a5f70aa1": (1, 1, 2, 2, 2, "how-to on Watch raise-to-wake; intake is safe but not an answer"),
    "c013bf8c9444": (0, 1, 2, 2, 1, "new Apple ID without a card is not account recovery"),
    "997726865f9b": (1, 0, 2, 2, 2, "missing order email; asking for a device model is ungrounded"),
    "722caba0215f": (1, 1, 2, 2, 1, "informational sign-out question framed as lockout"),
    "9d9c00b02492": (1, 1, 2, 2, 1, "classic Control Center Wi-Fi behavior, missed the public explainer"),
    "41eb70eaff90": (1, 1, 2, 2, 2, "iPhone X screenshot is a how-to Apple answered publicly"),
    "25571abdaf8c": (2, 2, 2, 2, 2, "correct keyboard-bug playbook"),
    "29d16946e2d1": (2, 2, 2, 2, 2, "correct language portal"),
    "f445a6bec940": (2, 2, 2, 2, 2, "correct keyboard-bug playbook"),
    "bbf50ab504d0": (2, 2, 2, 2, 2, "correct language portal"),
    "cc4339d72c83": (2, 2, 2, 2, 2, "correct language portal"),
    "4c1d6a4ad4db": (2, 2, 2, 2, 2, "correct Control Center explainer"),
    "114d9a55c9af": (2, 2, 2, 2, 2, "correct language portal"),
    "c4598a782e6c": (2, 2, 2, 2, 2, "correct language portal"),
    "65469cb0ef95": (2, 2, 2, 2, 2, "correct keyboard-bug playbook"),
    "aba0b66ae8e7": (2, 2, 2, 2, 2, "correct keyboard-bug playbook"),
    "3f627e1a3f34": (2, 2, 2, 2, 2, "correct battery intake"),
    "7abde7fdca23": (2, 2, 2, 2, 2, "correct battery intake on a multi-symptom tweet"),
    "af1f38dd4549": (2, 2, 2, 2, 2, "already-tried App Store case, correctly escalated"),
    "3da61e319784": (2, 2, 2, 2, 2, "pending purchase, asks the right billing question"),
    "aea94778085d": (0, 1, 2, 2, 2, "upgrade-loop freeze, not cracked-screen hardware"),
    "0bae8175d713": (1, 1, 2, 2, 2, "screenshot lock screen; draft is vague but correctly escalates"),
    "9d54840d0c5d": (2, 2, 2, 2, 2, "beta notification bug, correct intake"),
    "a334701da23e": (1, 1, 2, 2, 2, "misses named symptoms but escalates"),
    "e978d716386b": (2, 1, 2, 2, 2, "lost AirPods case; should ask country, still right to escalate"),
    "0470e676dba4": (1, 2, 2, 2, 2, "this is account recovery more than photo-library loss"),
    "bc2892b5294d": (1, 1, 2, 2, 2, "iTunes on Mac, draft still talks like iOS"),
    "88948117ac9a": (1, 1, 2, 2, 2, "contacts gone; draft never names contacts"),
    "68d4edd33532": (2, 2, 2, 2, 2, "correct battery intake"),
    "dc363ce77606": (1, 1, 2, 2, 2, "911 Watch event deserved an SOS-specific next step"),
    "325223dc20e9": (1, 1, 2, 1, 2, "heat/bulge is not ordinary drain; should warn about service"),
    "fde599d1f9f7": (2, 2, 2, 2, 2, "correct billing intake"),
    "e61b8ca89488": (2, 1, 2, 2, 2, "right to escalate; missing Safe Mode/Photos specialty"),
    "0abed9478783": (2, 2, 2, 2, 2, "correct battery intake"),
    "eb1aa3b22d44": (1, 1, 2, 2, 2, "4-day-old keyboard defect is hardware/service, not macOS tips"),
    "f7450e8dabdf": (2, 2, 2, 2, 2, "correct crash-loop intake"),
}


def main() -> None:
    preds = {}
    path = RESULTS_DIR / "golden_predictions.jsonl"
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            preds[r["id"]] = r
    rows = []
    missing = [i for i in HUMAN if i not in preds]
    if missing:
        raise SystemExit(f"missing predictions for {missing}")
    for i, (g, h, v, s, r, note) in HUMAN.items():
        p = preds[i]
        js = p["judge"]
        rows.append(
            {
                "id": i,
                "text": p["text"][:240],
                "reply": p["reply"],
                "groundedness": g,
                "helpfulness": h,
                "brand_voice": v,
                "safety": s,
                "routing_consistency": r,
                "rationale": note,
                "backend": "human",
                "auto_groundedness": js["groundedness"],
                "auto_helpfulness": js["helpfulness"],
                "auto_brand_voice": js["brand_voice"],
                "auto_safety": js["safety"],
                "auto_routing_consistency": js["routing_consistency"],
            }
        )
    HUMAN_JUDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HUMAN_JUDGE_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", len(rows), HUMAN_JUDGE_PATH)


if __name__ == "__main__":
    main()
