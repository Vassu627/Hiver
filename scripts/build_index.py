"""Build the retrieval corpus + intent model. Golden IDs are held out."""

from __future__ import annotations

import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apple_agent.config import CORPUS_PATH, INDEX_DIR, INTENTS, RAW_CSV
from apple_agent.intents import IntentClassifier, weak_intent
from apple_agent.retrieve import TfidfRetriever
from apple_agent.textutil import uid
from apple_agent.evaluate import load_jsonl
from apple_agent.config import GOLDEN_PATH


def load_golden_ids() -> set[str]:
    if not GOLDEN_PATH.exists():
        return set()
    return {r["id"] for r in load_jsonl(GOLDEN_PATH)}


def main() -> None:
    if not RAW_CSV.exists():
        raise SystemExit(f"Missing {RAW_CSV}. Run: python scripts/download_data.py")

    print("Loading", RAW_CSV)
    df = pd.read_csv(RAW_CSV)
    df = df.drop_duplicates("text_input").copy()
    df["id"] = df["text_input"].map(uid)
    df["weak_intent"] = df["text_input"].map(weak_intent)
    holdout = load_golden_ids()
    print("golden holdout", len(holdout))
    train_df = df[~df["id"].isin(holdout)]
    print("train rows", len(train_df))
    print(train_df["weak_intent"].value_counts().to_string())

    print("Training intent classifier on weak labels…")
    rng = random.Random(7)
    # Cap training size so index rebuild stays inside the 15-minute budget.
    train_cap = 28000
    train_rows = train_df.sample(n=min(train_cap, len(train_df)), random_state=7)
    clf = IntentClassifier.train(
        train_rows["text_input"].tolist(), train_rows["weak_intent"].tolist()
    )
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    clf.save(INDEX_DIR / "intent.joblib")

    rng = random.Random(7)
    # Stratified retrieval corpus so rare intents aren't drowned by iOS-update tweets.
    caps = {i: 1400 for i in INTENTS}
    caps["other"] = 1800
    caps["software_update_bug"] = 1600
    caps["battery_power"] = 1400
    picked = []
    by = defaultdict(list)
    for rec in train_df.itertuples(index=False):
        by[rec.weak_intent].append(rec)
    for intent, rows in by.items():
        rng.shuffle(rows)
        for rec in rows[: caps.get(intent, 800)]:
            picked.append(
                {
                    "id": rec.id,
                    "text": rec.text_input,
                    "apple_reply": rec.text_response,
                    "weak_intent": rec.weak_intent,
                }
            )
    rng.shuffle(picked)
    print("retrieval corpus", len(picked))
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(CORPUS_PATH, "wt", encoding="utf-8") as f:
        for row in picked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Fitting TF-IDF retriever…")
    retr = TfidfRetriever.build(picked)
    retr.save(INDEX_DIR)
    print("Wrote", INDEX_DIR)


if __name__ == "__main__":
    main()
