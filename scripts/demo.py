"""Interactive / one-shot demo of the Apple Support agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apple_agent.agent import SupportAgent
from apple_agent.config import INDEX_DIR
from apple_agent.intents import IntentClassifier
from apple_agent.retrieve import TfidfRetriever


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--message", required=True, help="Incoming customer message")
    args = p.parse_args()
    clf = IntentClassifier.load(INDEX_DIR / "intent.joblib")
    retr = TfidfRetriever.load(INDEX_DIR)
    agent = SupportAgent(clf, retr)
    result = agent.handle(args.message)
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
