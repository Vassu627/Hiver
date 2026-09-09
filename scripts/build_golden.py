"""Join hand labels with source tweets to write data/golden/golden.jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apple_agent.config import GOLDEN_PATH, RAW_CSV
from apple_agent.textutil import uid

LABELS_TSV = Path(__file__).resolve().parents[1] / "data" / "golden" / "labels.tsv"

REASONS = {
    "keyboard_public_workaround": "Known iOS 11 I/question-mark bug. Apple repeatedly sent a public workaround; a first reply can auto-send that article.",
    "language_portal": "Non-English. Apple's historical public move was a canned language-portal redirect, not in-thread troubleshooting.",
    "phishing_public_article": "Customer is asking whether a message/call is real. Apple answered with a public phishing explainer; no account mutation required.",
    "howto_public": "Complete how-to with a documented settings path or public article. Safe to auto-send a first-line answer.",
    "control_center_explainer": "Wi-Fi/Bluetooth Control Center 'doesn't really turn off' is expected iOS 11 behavior Apple explained publicly.",
    "feature_feedback": "Feature request / comparison question, not a broken device. Apple pointed at the feedback page or a spec sheet.",
    "account_recovery": "Apple ID, 2FA, Activation Lock, or disabled device. A wrong auto-reply can lock someone out; human-owned.",
    "billing_verify": "Money movement (charge, refund, pending purchase). Must be checked against the Apple ID by a human.",
    "data_loss": "Photos, notes, songs, or iCloud data may be gone. Do not auto-send wipes or guesses.",
    "hardware_service": "Physical damage or likely hardware/service. Needs region/warranty; Apple collected that privately.",
    "battery_intake": "Battery drain after an OS update was almost never closed in public. Needs model + iOS + a specialist.",
    "update_bug_intake": "Crashes/freezes/boot loops after an update need repro details. Public canned replies did not fix these.",
    "accessory_diagnostics": "Watch/AirPods hardware or pairing. Apple asked for watchOS/iOS versions or moved to DM.",
    "macos_diagnostics": "Mac install/performance/bricking. Often Safe Mode/NVRAM or a specialty team — not auto-send.",
    "insufficient_context": "Screenshot-only, mid-thread fragment, or too short to act on without the rest of the conversation.",
    "already_tried": "Customer already exhausted first-line steps. Another canned loop should not auto-send.",
    "sales_pointer": "Sales/fulfillment (case, order email, upgrade program). Public pointer to Online Sales is enough to auto-send.",
    "close_resolved": "Customer is confirming a fix or saying thanks. A short public close is fine to auto-send.",
}


def main() -> None:
    labels = pd.read_csv(LABELS_TSV, sep="\t")
    df = pd.read_csv(RAW_CSV)
    df = df.drop_duplicates("text_input").copy()
    df["id"] = df["text_input"].map(uid)
    merged = labels.merge(df, on="id", how="left")
    missing = merged[merged["text_input"].isna()]
    if len(missing):
        raise SystemExit(f"missing ids: {missing['id'].tolist()[:20]}")
    rows = []
    for rec in merged.itertuples(index=False):
        rows.append(
            {
                "id": rec.id,
                "text": rec.text_input,
                "intent": rec.intent,
                "route": rec.route,
                "route_reason": REASONS[rec.reason_tag],
                "reason_tag": rec.reason_tag,
                "notes": rec.notes if isinstance(rec.notes, str) else "",
                "apple_reply": rec.text_response,
                "apple_asked_dm": bool(
                    pd.notna(rec.text_response)
                    and (
                        " dm " in f" {rec.text_response.lower()} "
                        or "direct message" in rec.text_response.lower()
                    )
                ),
            }
        )
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "golden examples")
    print(pd.Series([r["intent"] for r in rows]).value_counts().to_string())
    print("route", pd.Series([r["route"] for r in rows]).value_counts().to_dict())


if __name__ == "__main__":
    main()
