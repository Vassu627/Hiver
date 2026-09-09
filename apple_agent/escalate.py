from __future__ import annotations

import re
from dataclasses import dataclass

from apple_agent import textutil as T
from apple_agent.intents import IntentPrediction


@dataclass
class RouteDecision:
    route: str  # auto | escalate
    reason: str
    score: int
    signals: list[str]


MONEY = re.compile(r"\$\d|\border id\b|refund|charged|purchase history", re.I)
DATA_LOSS = re.compile(
    r"(all (my )?(photos|songs|notes|contacts|videos).{0,20}(gone|missing|wiped)|"
    r"(photos|songs|notes|contacts|files).{0,24}(gone|missing|wiped)|"
    r"lost (my )?(photos|pictures|notes|data)|"
    r"won'?t download|disappeared|wiped when)",
    re.I,
)
CHARACTER_BUG = re.compile(
    r"(I️|question mark|weird char|boxed \?|letter [\"']i[\"']|autocorrect|I\.T|Illumin)",
    re.I,
)
PHISHING_ASK = re.compile(
    r"is this (real|spam|a scam)|phishing|real or spam", re.I
)
CONTROL_CENTER = re.compile(
    r"control (center|panel|centre)|turn(ing)? off wifi|wifi widget|"
    r"doesn't really turn off|does not turn off",
    re.I,
)
FEATURE_REQUEST = re.compile(
    r"wish u guys|feedback|suggestion|when are y.?all going to fix the letter",
    re.I,
)
SALES = re.compile(
    r"upgrade program|hard case|order .{0,12}email|online store|beats x or it is just not",
    re.I,
)


def decide(
    text: str,
    intent_pred: IntentPrediction,
    retrieval_score: float,
) -> RouteDecision:
    """Default is escalate. Auto only with an explicit public-playbook allowlist and no veto."""
    intent = intent_pred.intent
    signals: list[str] = []

    if T.looks_non_english(text) or intent == "language_unsupported":
        return RouteDecision(
            route="auto",
            reason="Non-English request: Apple historically routed these to the language portal rather than troubleshooting in-thread.",
            score=-3,
            signals=["language_redirect"],
        )

    vetoes = _vetoes(text, intent)
    signals.extend(vetoes)

    allow, allow_reason, allow_sig = _allowlist(text, intent)
    if allow_sig:
        signals.append(allow_sig)

    if vetoes:
        return RouteDecision(
            route="escalate",
            reason=_veto_reason(vetoes, intent),
            score=3 + len(vetoes),
            signals=signals,
        )

    if allow:
        return RouteDecision(
            route="auto",
            reason=allow_reason,
            score=-2,
            signals=signals,
        )

    # Default: human reviews. Extra score if the message is thin or off-corpus.
    score = 2
    if T.is_screenshot_only(text) or T.is_too_short(text):
        signals.append("insufficient_context")
        score += 1
    if retrieval_score < 0.10:
        signals.append("low_retrieval")
        score += 1
    if intent == "other":
        signals.append("unclear_intent")
    return RouteDecision(
        route="escalate",
        reason=_default_reason(intent, signals),
        score=score,
        signals=signals,
    )


def _vetoes(text: str, intent: str) -> list[str]:
    v = []
    if intent in {"account_security", "hardware", "macos"} or re.search(
        r"activation lock|locked out|verification code|apple id .{0,12}disabled|hacked",
        text,
        re.I,
    ):
        # phishing *ask* is handled as allow unless money/account mutation
        if not PHISHING_ASK.search(text):
            if intent == "account_security" or re.search(
                r"activation lock|locked out|verification code|disabled", text, re.I
            ):
                v.append("account_stakes")
            if intent in {"hardware", "macos"} and not SALES.search(text):
                v.append(f"high_stakes:{intent}")
            if re.search(r"\bhacked\b", text, re.I) and not PHISHING_ASK.search(text):
                v.append("account_stakes")

    if MONEY.search(text) or intent == "billing_purchases":
        if not (PHISHING_ASK.search(text) and not MONEY.search(text)):
            v.append("money")
    if intent == "icloud_data" or DATA_LOSS.search(text):
        v.append("data_loss")
    if intent == "battery_power" or re.search(
        r"batter|dying|drain|overheat|won'?t charge", text, re.I
    ):
        # battery how-to for *showing percentage* is allowlisted separately
        if not re.search(r"battery %|battery percent|see my remaining battery", text, re.I):
            v.append("battery")
    if T.REPEAT_HELP.search(text):
        v.append("already_tried")
    if T.LEGAL.search(text):
        v.append("legal")
    if re.search(r"\b911\b|hot i was afraid|bulge", text, re.I):
        v.append("safety_event")
    if intent == "accessories" and not FEATURE_REQUEST.search(text):
        # spec comparison can auto; hardware/pairing cannot
        if not re.search(r"equally made or different|compare", text, re.I):
            v.append("accessory")
    if intent == "software_update_bug":
        v.append("device_bug")
    if intent == "connectivity" and not CONTROL_CENTER.search(text):
        v.append("connectivity_persistent")
    # keyboard character bug that already got the 11.1.1 fix
    if CHARACTER_BUG.search(text) and re.search(r"11\.1\.1", text) and re.search(
        r"still|keeps appearing|yes, i updated", text, re.I
    ):
        v.append("already_tried")
    return v


def _allowlist(text: str, intent: str) -> tuple[bool, str, str]:
    if CHARACTER_BUG.search(text) or T.is_keyboard_bug(text):
        if not re.search(r"keyboard freeze|not an autocorrect", text, re.I):
            return (
                True,
                "Known public workaround cluster (I/question-mark keyboard bug). Apple repeatedly sent the same article without a DM.",
                "keyboard_workaround",
            )
    if PHISHING_ASK.search(text) and not MONEY.search(text):
        return (
            True,
            "Customer is asking to authenticate a likely phishing message. Apple historically answered this with a public phishing article.",
            "phishing_education",
        )
    if CONTROL_CENTER.search(text):
        return (
            True,
            "Wi-Fi/Bluetooth Control Center behavior was explained with the same public article.",
            "known_control_center_behavior",
        )
    if intent == "how_to_feature" or re.search(
        r"how (do i|can i|do you)|where (do|can) i", text, re.I
    ):
        if not DATA_LOSS.search(text) and not re.search(r"batter|dying|crash", text, re.I):
            return (
                True,
                "A first-contact how-to with a documented setting path can be sent without a specialist.",
                "public_howto",
            )
    if FEATURE_REQUEST.search(text) or intent == "how_to_feature" and re.search(
        r"wish|suggestion|feedback", text, re.I
    ):
        return (
            True,
            "Feature request. Apple closed similar tweets with the public feedback page.",
            "feature_feedback",
        )
    if SALES.search(text):
        return (
            True,
            "Sales/fulfillment question. Apple historically pointed at Online Sales in public.",
            "sales_pointer",
        )
    return False, "", ""


def _veto_reason(vetoes: list[str], intent: str) -> str:
    if "safety_event" in vetoes:
        return "Safety-adjacent (overheating/911). A human must own the next step."
    if "account_stakes" in vetoes:
        return "Account lockout/recovery can lock someone out of a device. Apple historically moved these to account specialists — a human must own it."
    if "money" in vetoes:
        return "Charges, refunds, and store purchases are account-specific. Auto-sending a generic reply risks the wrong financial action."
    if "data_loss" in vetoes:
        return "Possible data loss. Historical Apple replies asked for library/backup details in a private channel."
    if "already_tried" in vetoes:
        return "Customer already exhausted first-line steps. Another canned troubleshooting loop would waste the thread."
    if "battery" in vetoes:
        return "Battery drain after an OS update was Apple's most common DM case. Public tweets rarely resolved it."
    if "high_stakes:hardware" in vetoes:
        return "Physical damage or device-service questions need region and warranty, which Apple collected privately."
    if "high_stakes:macos" in vetoes:
        return "Mac install/bricking needs Safe Mode/NVRAM or a specialty team — not an auto-send."
    if "accessory" in vetoes:
        return "Watch/AirPods hardware or pairing. Apple asked for OS versions or moved to DM."
    if "device_bug" in vetoes:
        return "Crash/freeze/update regressions need model + iOS version + repro, which is a human-owned intake."
    if "connectivity_persistent" in vetoes:
        return "Persistent radio issues are not the Control Center explainer. Apple usually collected diagnostics."
    return "Issue is not covered by a public, deterministic playbook, so a human should review the draft before send."


def _default_reason(intent: str, signals: list[str]) -> str:
    if "insufficient_context" in signals:
        return "Message is a screenshot, a mid-thread fragment, or too short to act on without the rest of the conversation."
    return "Issue is not covered by a public, deterministic playbook, so a human should review the draft before send."
