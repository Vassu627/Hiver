from __future__ import annotations

import re

from apple_agent import textutil as T
from apple_agent.escalate import RouteDecision
from apple_agent.intents import IntentPrediction

# Public article *types* Apple actually used in this corpus (2017 iOS 11 era).
# We copy URLs only from retrieved Apple replies so drafts stay historically grounded.

VOICE_OPENERS = {
    "calm": "We're here to help.",
    "empathy": "That's not the experience we want you to have.",
    "thanks": "Thanks for reaching out.",
    "got_it": "Got it — thanks for the detail.",
}

INTENT_ACK = {
    "keyboard_input": "This looks like the iOS keyboard/character bug a lot of people hit around this software release.",
    "battery_power": "Battery drain like that isn't what we expect from the device.",
    "software_update_bug": "A software update shouldn't leave the phone unstable like that.",
    "connectivity": "Let's sort out the Wi-Fi/Bluetooth behavior.",
    "hardware": "We can look at the hardware/service options for that.",
    "icloud_data": "Your photos, backups, and iCloud data are important — we don't want to guess here.",
    "billing_purchases": "Charges and store purchases need to be checked against the actual account.",
    "account_security": "Apple ID / lockout issues need the account-security path, not a public workaround.",
    "accessories": "We can take a look at the Watch/AirPods setup.",
    "macos": "We can look at the Mac-side steps for this.",
    "how_to_feature": "That's a fair how-to question.",
    "language_unsupported": "We offer Twitter support in English.",
    "other": "We want to make sure we understand the issue before we change anything on the device.",
}


def _article_url(neighbors: list[dict]) -> str | None:
    """Prefer public-article links; skip Apple's generic 'DM us' shortlink."""
    dm_only = []
    for nb in neighbors[:8]:
        reply = nb.get("apple_reply") or ""
        urls = T.extract_urls(reply)
        if not urls:
            continue
        is_dm = bool(re.search(r"\bdm\b|direct message", reply, re.I))
        is_article = bool(
            re.search(r"article|workaround|steps here|check out|this link", reply, re.I)
        )
        if is_article or not is_dm:
            return urls[0]
        dm_only.extend(urls)
    return None


def _apple_style_flags(neighbors: list[dict]) -> dict[str, bool]:
    replies = " ".join((nb.get("apple_reply") or "").lower() for nb in neighbors[:5])
    return {
        "asked_dm": bool(re.search(r"\bdm\b|direct message", replies)),
        "asked_version": bool(re.search(r"ios version|settings > general > about", replies)),
        "workaround": bool(re.search(r"work around|workaround", replies)),
        "language": bool(re.search(r"support via twitter in english", replies)),
        "phishing": bool(re.search(r"phishing", replies)),
        "feedback": bool(re.search(r"feedback page", replies)),
        "iforgot": bool(re.search(r"iforgot|apple id issue|account security", replies)),
        "article": bool(re.search(r"https://t.co/", replies)),
    }


def draft_reply(
    text: str,
    intent_pred: IntentPrediction,
    route: RouteDecision,
    neighbors: list[dict],
) -> str:
    intent = intent_pred.intent
    flags = _apple_style_flags(neighbors)
    device = T.extract_device(text)
    if device:
        device = device.strip().title().replace("Ios", "iOS").replace("Iphone", "iPhone").replace("Ipad", "iPad").replace("Iphone ", "iPhone ")
    ios = T.extract_ios(text)
    opener = VOICE_OPENERS["empathy"] if T.anger_score(text) else VOICE_OPENERS["calm"]
    ack = INTENT_ACK.get(intent, INTENT_ACK["other"])
    url = _article_url(neighbors)

    if "language_redirect" in route.signals or intent == "language_unsupported":
        body = (
            "We offer support via Twitter in English. "
            "Contact us in your preferred language through Apple Support online "
            "(the same language-routing reply Apple used on this channel)."
        )
        if url:
            body += f" {url}"
        return body

    if "phishing_education" in route.signals:
        body = (
            f"{opener} You are right to question that message. "
            "Apple will not ask you to confirm your Apple ID over a random call or unpaid invoice tweet. "
            "Treat it as phishing and do not tap links or give codes."
        )
        if url:
            body += f" {url}"
        return body

    if intent == "keyboard_input" or "keyboard_workaround" in route.signals:
        body = (
            f"{ack} Apple published a public workaround for this and repeatedly pointed people to it "
            "while a software fix rolled out. Back up the device, then apply the workaround in that article"
        )
        if url:
            body += f": {url}."
        else:
            body += " (keyboard/text replacement workaround Apple Support linked in similar threads)."
        if route.route == "escalate":
            body += " If you already tried that, we'll have a specialist take the next step — reply with your exact iOS version (Settings > General > About)."
        else:
            body += " If it is still happening after that, reply with your device and iOS version."
        return _clip(body)

    if "known_control_center_behavior" in route.signals:
        body = (
            f"{opener} In iOS 11, Control Center disconnects Wi-Fi/Bluetooth from the current accessory "
            "but does not fully power the radios off — that's why the icon can still look on. "
            "To fully disable them, use Settings. Apple explained this with the same public article on similar tweets."
        )
        if url:
            body += f" {url}"
        return _clip(body)

    parts = [f"{opener} {ack}"]

    if device:
        parts.append(f"We see this is on {device}" + (f" / iOS {ios}." if ios else "."))
    elif ios:
        parts.append(f"Thanks for the iOS {ios} detail.")

    if route.route == "auto":
        parts.append(_auto_next_step(intent, url, flags))
    else:
        parts.append(_escalation_intake(intent, url, flags, device, ios))

    return _clip(" ".join(p for p in parts if p))


def _auto_next_step(intent: str, url: str | None, flags: dict[str, bool]) -> str:
    if intent == "how_to_feature":
        step = (
            "Here's the first-line path Apple usually started with on similar how-tos: "
            "check Settings > General > About for the exact iOS version, then try the matching Support article for that feature."
        )
        if url:
            step += f" Closest historical article: {url}"
        return step
    if flags["feedback"]:
        return "The best way to get a feature change on the record is Apple's product feedback page — that's how Apple closed similar request-only tweets."
    if url:
        return f"On similar threads Apple sent this public article rather than opening a private ticket first: {url}"
    return "Try the public first-line step Apple used for this class of issue, then reply if it still fails."


def _escalation_intake(
    intent: str,
    url: str | None,
    flags: dict[str, bool],
    device: str | None,
    ios: str | None,
) -> str:
    ask = []
    if not device:
        ask.append("device model")
    if not ios and intent in {
        "battery_power",
        "software_update_bug",
        "connectivity",
        "hardware",
        "keyboard_input",
    }:
        ask.append("exact iOS version (Settings > General > About)")
    if intent == "account_security":
        extra = (
            "Do not post your password or verification codes here. "
            "A specialist should take Apple ID recovery (iforgot / account security), which is what Apple did historically."
        )
        if url:
            extra += f" Start with: {url}"
        return extra
    if intent == "billing_purchases":
        extra = (
            "A human should check the actual purchase/refund record. "
            "Reply with whether the charge appeared on the Apple ID purchase history or only on a bank SMS/email — Apple asked that before acting."
        )
        if url:
            extra += f" {url}"
        return extra
    if intent == "icloud_data":
        extra = (
            "Because this may be data loss, we should not auto-send a reset. "
            "A specialist will ask whether iCloud Photo Library / backup was on, and whether the items still appear in Moments or on icloud.com."
        )
        return extra
    if intent == "hardware":
        extra = (
            "Service options depend on region and whether this is physical damage vs. a software display glitch. "
            "Please tell us the country you're in and whether the device has any cracks or liquid contact."
        )
        return extra
    intake = "This needs a human-owned next step rather than an auto-send."
    if ask:
        intake += " To start, send " + " and ".join(ask) + "."
    elif flags["asked_version"]:
        intake += " Apple usually asked for the exact iOS version from Settings > General > About before going further."
    if url and flags["article"]:
        intake += f" Related article from similar resolved threads: {url}"
    return intake


def _clip(text: str, limit: int = 520) -> str:
    text = T.WS.sub(" ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"
