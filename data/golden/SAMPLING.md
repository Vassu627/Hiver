# Golden set sampling and labelling

**n = 231** customer tweets to Apple Support, labelled by hand.

## Population

Source is the Apple-filtered slice of *Customer Support on Twitter* (Thought Vector / Kaggle), distributed as `OpenArchive/AppleConvos` (106,648 inbound→Apple reply pairs). Each row is already a customer message + first Apple reply. User IDs in the public file are hashed.

## How examples were sampled

1. Deduplicate inbound text.
2. Weak-label with the same keyword rules later used as features (not as gold).
3. Stratified draw with a fixed seed (`42`) so rare classes (account lockout, AirPods, non-English) are not drowned by iOS-11 battery tweets.
4. Mix of Apple-asked-DM and public-article replies inside each stratum.
5. **Dropped 19 candidates** that were not standalone: screenshot-only with no words, mid-thread fragments (`iPhone 7 iOS 11`, `6 giant`), or thank-you closes. Those are a real production problem — they are discussed in the report as a limitation, not silently scored.

The labelled IDs are held out of the retrieval corpus and the intent-model training set.

## What was labelled (independently of the agent)

| Field | Meaning |
|---|---|
| `intent` | One of 13 intents defined from the data, not from Banking77. |
| `route` | `auto` = I would let this draft send without a human. `escalate` = a human must review. |
| `route_reason` | Why. This is a **trust judgement**, not “did Apple ask for a DM?” |
| `apple_reply` | Historical reply, stored as evidence / a misleading baseline target, not as the only correct draft. |

Labelling took one pass over the 250-candidate file plus a second pass on every `auto` example (false auto is the expensive error). Disagreements with Apple’s own DM habit were kept on purpose: Apple DMed ~53% of everything, including cases where they also had a public article. Gold follows the **inbox-agent** policy in the README, not Twitter-era DM culture.

## Intent mix (gold)

See `labels.tsv`. Roughly: software-update and battery are the head; account, iCloud, accessories, keyboard, language, connectivity are the torso; billing/hardware/macos/how-to are the tail. `other` is reserved for genuinely underspecified rants and phishing-check tweets that are not a device issue.

## Known labelling caveats

- The corpus is **2017 / iOS 11**. “Good” replies are historically grounded, not 2026 AppleCare policy.
- Several tweets are turn 2+ of a thread. If the inbound is still a complete question, it was kept; if it is only context for a missing tweet, it was dropped or tagged `insufficient_context`.
- Dual-intent tweets (battery + crash after an update) were given the **stakes-higher** label (usually battery or data loss), not the first keyword hit.
