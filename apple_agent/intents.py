from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from apple_agent.config import INTENTS
from apple_agent import textutil as T

INTENT_SET = set(INTENTS)

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "account_security",
        re.compile(
            r"apple ?id|locked out|activation lock|verification code|"
            r"two[- ]factor|\b2fa\b|password reset|hacked|disabled.{0,20}id|"
            r"iforgot|security questions|stolen (iphone|phone|device)",
            re.I,
        ),
    ),
    (
        "billing_purchases",
        re.compile(
            r"charged|refund|subscription|app store|billing|double billed|"
            r"unauthorized|purchase history|family share|apple music|"
            r"itunes store|payment (method|details)|beats .*(money|refund)",
            re.I,
        ),
    ),
    (
        "icloud_data",
        re.compile(
            r"icloud|lost (my )?(photos|pictures|contacts|notes|data)|"
            r"photo library|icloud drive|backup (failed|photos)|"
            r"wiped when i did the ios|songs .{0,40}(gone|missing)|"
            r"contacts .{0,20}gone|photos .{0,20}gone",
            re.I,
        ),
    ),
    (
        "accessories",
        re.compile(r"airpods|apple watch|watchos|homepod|digital crown", re.I),
    ),
    (
        "macos",
        re.compile(
            r"macbook|\bimac\b|mac mini|macos|mac os|high sierra|"
            r"el capitan|nvram|finder",
            re.I,
        ),
    ),
    (
        "keyboard_input",
        re.compile(
            r"autocorrect|auto correct|question mark|I\.T|weird char|boxed \?|I️|"
            r"letter [\"']i[\"']|Illumin",
            re.I,
        ),
    ),
    (
        "battery_power",
        re.compile(
            r"batter|won'?t charge|not charging|drain|dies at|died at|"
            r"overheat|hot i was afraid|charge\?|charging cables",
            re.I,
        ),
    ),
    (
        "connectivity",
        re.compile(
            r"wi-?fi|wifi|bluetooth|cellular|no service|no signal|"
            r"control center|control panel|airdrop",
            re.I,
        ),
    ),
    (
        "hardware",
        re.compile(
            r"screen (cracked|protector)|cracked|speaker|home button|"
            r"touch id|face id|camera|display issue|digitizer|"
            r"blank screen|streaks are left|discolored",
            re.I,
        ),
    ),
    (
        "software_update_bug",
        re.compile(
            r"ios\s*\d|after (the )?update|new update|latest update|"
            r"software update|keeps (crashing|restarting|freezing)|"
            r"boot loop|won'?t turn on|factory reset",
            re.I,
        ),
    ),
    (
        "how_to_feature",
        re.compile(r"how (do i|can i|do you)|where (do|can) i|how i (can|do)", re.I),
    ),
]


def weak_intent(text: str) -> str:
    if T.looks_non_english(text):
        return "language_unsupported"
    if T.is_keyboard_bug(text):
        return "keyboard_input"
    # account/billing before generic "update"
    for name, pat in _PATTERNS:
        if name in {"software_update_bug", "how_to_feature"}:
            continue
        if pat.search(text):
            return name
    if _PATTERNS[-2][1].search(text):  # software_update_bug
        return "software_update_bug"
    if _PATTERNS[-1][1].search(text):
        return "how_to_feature"
    return "other"


@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    weak_intent: str
    scores: dict[str, float]


class IntentClassifier:
    """TF-IDF + logistic regression trained on weak labels, with a rule override."""

    def __init__(self, sklearn_model=None, vectorizer=None):
        self.model = sklearn_model
        self.vectorizer = vectorizer

    def predict(self, text: str) -> IntentPrediction:
        weak = weak_intent(text)
        scores = {i: 0.0 for i in INTENTS}
        scores[weak] = 0.55
        conf = 0.55
        intent = weak

        if self.model is not None and self.vectorizer is not None:
            x = self.vectorizer.transform([T.normalize(text)])
            proba = self.model.predict_proba(x)[0]
            mapping = {c: float(p) for c, p in zip(self.model.classes_, proba)}
            for i in INTENTS:
                scores[i] = mapping.get(i, 0.0)
            ml_intent = max(scores, key=scores.get)
            ml_conf = scores[ml_intent]
            # High-precision rules win on distinctive / high-stakes classes
            if weak in {
                "account_security",
                "language_unsupported",
                "keyboard_input",
                "billing_purchases",
                "battery_power",
                "icloud_data",
                "hardware",
                "accessories",
            }:
                intent, conf = weak, max(0.72, ml_conf)
            elif ml_conf >= 0.38:
                intent, conf = ml_intent, ml_conf
            else:
                intent, conf = weak, max(ml_conf, 0.4)
            scores[weak] = max(scores.get(weak, 0), 0.15)

        return IntentPrediction(
            intent=intent if intent in INTENT_SET else "other",
            confidence=float(conf),
            weak_intent=weak,
            scores=scores,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "vectorizer": self.vectorizer}, path)

    @classmethod
    def load(cls, path: Path) -> "IntentClassifier":
        blob = joblib.load(path)
        return cls(sklearn_model=blob["model"], vectorizer=blob["vectorizer"])

    @classmethod
    def train(cls, texts: list[str], labels: list[str]) -> "IntentClassifier":
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_features=50000,
            sublinear_tf=True,
        )
        x = vec.fit_transform([T.normalize(t) for t in texts])
        clf = LogisticRegression(
            max_iter=250,
            class_weight="balanced",
            C=2.0,
        )
        clf.fit(x, labels)
        return cls(sklearn_model=clf, vectorizer=vec)
