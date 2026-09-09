# Apple Support agent — Hiver SDE intern take-home

An inbox agent for **Apple Support**, trained and evaluated on real Twitter threads. It classifies an inbound customer message, drafts a reply grounded in how Apple actually closed similar cases, and decides **auto-send vs escalate** with a reason.

The number to look at first is not reply similarity. It is **false auto-sends on a 231-example hand-labelled set: 2 / 165 escalate cases (1.2%)**, vs 88 for a retrieve-and-copy baseline.

Snapshot (also written to `results/metrics.json` when you run eval):

| System | Intent acc / macro-F1 | Route acc | Escalate recall | False autos | TF-IDF cosine vs Apple's tweet | Judge mean /10 | Pass@7 | Pass@9 |
|---|---|---|---|---|---|---|---|---|
| Trivial (always `other` + escalate + canned sorry) | 0.039 / 0.006 | 0.714 | **1.000** | **0** | 0.047 | 6.00 | 0.00 | 0.00 |
| Simple (TF-IDF nearest neighbor, copy Apple's reply) | 0.485 / 0.485 | 0.545 | 0.467 | 88 | **0.119** | 7.43 | 0.75 | 0.26 |
| **This agent** | **0.788 / 0.773** | **0.909** | 0.988 | **2** | 0.072 | 9.68 | 1.00 | 0.93 |

Judge–human study (n=40, same rubric): pass@7 agreement **0.90**; exact dimension agreement voice 1.00 / safety 0.98 / routing 0.78 / helpfulness 0.58 / groundedness 0.53. Quadratic-weighted kappa on the 0–10 total is ~0 — see [What's misleading](#what-is-misleading-about-my-headline-number).

---

## Reproduce the headline table (< 15 minutes)

Python 3.10+ . No API key required. From the repo root:

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py          # ~27MB Apple-filtered CSV from Hugging Face
python scripts/build_index.py            # hold out golden IDs; fit TF-IDF + logistic regression
python scripts/run_eval.py               # prints the table above; writes results/metrics.json
```

Optional:

```bash
python scripts/demo.py --message "my iPhone battery dies at 30% after the iOS update"
# LLM-as-judge backend (same rubric). Headline numbers use the programmatic backend.
set OPENAI_API_KEY=...   # Windows
python scripts/run_eval.py --judge openai
```

Rebuild the golden JSONL from the hand labels (only if you change `data/golden/labels.tsv`):

```bash
python scripts/build_golden.py
```

Data source: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Thought Vector), Apple-only slice [OpenArchive/AppleConvos](https://huggingface.co/datasets/OpenArchive/AppleConvos) (106,648 pairs, CC BY-NC-SA 4.0). Same schema as the Kaggle CSV, already paired inbound → Apple reply.

---

## Problem framing: what “good” means, and what I did not build

Hiver’s product is an **inbox copilot**, not a public Twitter bot. For this brand, a good system is one a lead would let sit in front of AppleCare:

1. **Intent** is specific enough to pick a playbook (battery vs Apple ID vs “the I character is broken”).
2. **Route** is conservative. A false auto-send on lockout, money, or data loss is worse than extra escalations. Apple’s own channel DMed ~53% of replies; blindly copying that is not a policy.
3. **Draft** is something a human can send or lightly edit: Apple’s calm voice, one next step, grounded in *how this class of issue was actually closed* (workaround article, language portal, Control Center explainer, or diagnostic intake). It must not ask for passwords in public or promise that an account is already unlocked.

**Not “good”:** matching 2017 Apple tweets token-for-token. Those tweets are full of “DM us” and “update to iOS 11.1.1”. Cosine-to-history is a baseline metric, not the objective.

### What I chose not to build

- **Multi-turn state.** The file is pairs, not reconstructed threads. Mid-thread fragments are labelled `insufficient_context` and should escalate. A real week-two item.
- **Banking77.** Apple intents do not transfer from bank-account NLU. Using it would inflate an intent number that does not apply here.
- **A live LLM in the default path.** Reviewers must reproduce in 15 minutes without keys. Generation is retrieve + playbook + Apple-voice templates. The LLM-as-judge *spec* is in `apple_agent/judge.py`; OpenAI is optional.
- **Tool-using AppleCare** (look up serial, book Genius Bar, issue refunds). That is the product; this assignment is the decision layer in front of it.
- **The other ~3M tweets / other brands.** One brand, evaluated hard, beats a multi-brand demo that nobody trusts.

---

## System

```
inbound tweet
    → weak rules + TF-IDF logistic intent (13 classes from this corpus)
    → TF-IDF retrieval over ~18k held-out historical (customer, Apple reply) pairs
    → allowlist router (default escalate; auto only for public playbooks, with vetoes)
    → draft: Apple voice + retrieved article URL if Apple used one, else intake questions
```

**Intents** (defined from frequency and stakes, not from a public taxonomy): `keyboard_input`, `battery_power`, `software_update_bug`, `connectivity`, `hardware`, `icloud_data`, `billing_purchases`, `account_security`, `accessories`, `macos`, `how_to_feature`, `language_unsupported`, `other`.

**Auto-send allowlist** (historically public Apple moves): iOS 11 I/question-mark workaround; non-English language portal; “is this email real?” phishing explainer; Control Center Wi-Fi/Bluetooth does-not-power-off explainer; complete how-tos; sales pointers; feature-feedback pointers.

**Vetoes** (always escalate): money, Apple ID lockout, data loss, battery drain, hardware/service, persistent radio issues, already-tried loops, 911/overheating, Mac bricking.

Golden IDs are **held out** of both the classifier training pool and the retrieval corpus.

---

## Golden set

231 examples. Sampling and labelling notes: [`data/golden/SAMPLING.md`](data/golden/SAMPLING.md). Hand labels: [`data/golden/labels.tsv`](data/golden/labels.tsv).

Route gold is a **human trust judgement** (“would I let this send?”), not “did Apple ask for a DM?” and not a dump of the router code. Mix: 165 escalate / 66 auto. That 29% auto rate is the product, not the historical DM rate.

---

## Results vs two baselines

**Trivial** always escalates. It therefore *wins* escalate-recall and false-auto (0). It also “wins” route accuracy on a 71% escalate set. It cannot classify, and the judge gives it 6/10 (no concrete next step). If the headline were only “escalate F1”, trivial looks competitive. That is why false-auto **and** intent **and** judge are reported together.

**Copy-nearest** is the serious baseline: embed (here: TF-IDF) the inbound, paste Apple’s historical reply, escalate iff that reply said “DM”. It **wins cosine-to-Apple (0.119 vs 0.072)**. It also auto-sends 88 cases a human labelled escalate, because Apple often posted a public-looking sentence that was still a brush-off, or because the neighbor was the keyboard workaround for a crash/battery tweet. Intent accuracy 48.5% is just the weak label of the neighbor.

**This agent** trades similarity for policy. Intent 78.8% / 77.3% macro-F1. Route 90.9%, escalate recall 98.8%, **2 false autos**. Judge pass@9 92.6% vs 26.4% copy-nearest vs 0% trivial.

Per-class F1 (agent): language 1.00, connectivity 0.94, account 0.92, battery 0.89, iCloud 0.87, keyboard 0.84, accessories 0.82, macos 0.75, how-to 0.69, software-update 0.68, hardware 0.67, billing 0.61, **other 0.36**. The tail is where the failure analysis lives.

---

## Failure analysis (top 5, with examples)

### 1. Screenshot-dependent how-tos auto-send (both false autos)

Gold: escalate (`insufficient_context`). Pred: auto how-to.

- Duplicate ringtones + image (`6af36d260e7b`): draft says “check Settings > General > About”, which does not remove ringtones.
- App switcher “showing apps **this way**” + image (`129d8c534c5a`): the content is in the screenshot. The agent cannot see it and still auto-sends.

Hypothesis: the how-to allowlist keys on `how (do|can) I` and ignores that the actual referent is an image. Fix: any inbound with a `t.co` image and no executable settings path in the text should veto auto.

### 2. `software_update_bug` recall is 0.56 — vague “iOS 11 ruined my phone” falls into `other`

Example: “Am I the only one with problems with new IOS release? Screen dims, screen shot’s can’t text / hung…” — gold `software_update_bug`, pred `other`. Route still escalates (default), so this is an **intent** miss, not a safety miss. Hypothesis: the weak-label prior for `other` is huge (~53k / 104k), and “update” alone is not a reliable n-gram once iOS version numbers are stripped for TF-IDF. Fix: a “post-update instability” pattern that does not require the word crash.

### 3. High-stakes how-tos are over-framed as lockout/data-loss (false escalate)

Gold auto, pred escalate:

- “If I sign out from my Apple ID, would I lose any data?”
- “I need to get a new apple id without using a credit card”
- “how do I delete game data… I’ve gone into iCloud storage”

Hypothesis: vetoes fire on `apple id` / `icloud` / `lost` before the allowlist how-to. That is the conservative bias I wanted for money/lockout; it also blocks informational questions Apple *did* answer publicly. Cost: extra human time, not a wrong send.

### 4. Dual-intent tweets take the keyword that is easiest, not the one with stakes

“Battery life sucks… keyboard freezes… apps constantly crash” is gold `battery_power` (highest-stakes public complaint in this corpus). An earlier router treated `keyboard` as the iOS 11 character bug and would have auto-sent the workaround article. The current veto on battery words stops the auto-send; intent can still wobble. Hypothesis: single-label gold is a lie for this dataset. Next week: multi-label + “max stakes wins the route”.

### 5. Drafts escalate correctly but talk about the wrong object

Upgrade-loop freeze (“press home button to upgrade”) predicted `hardware` and asked about cracks/liquid. 911 Watch SOS got a generic “Watch/AirPods setup” paragraph. Hypothesis: intake templates are keyed on predicted intent, not on retrieved Apple replies. When intent is wrong, the draft is confidently about the wrong playbook. Retrieval is used for URLs more than for *what to ask*.

---

## What is misleading about my headline number?

Several numbers look better than the system:

1. **Judge mean 9.68 / 10 and pass@7 = 100%.** The programmatic judge is a checklist over the same templates the agent emits (Apple voice, “specialist”, “Settings > General > About”). It saturates. Trivial fails (6.0) and copy-nearest is lower (7.4), so it *ranks* systems, but it does not mean a lead would rate these drafts 9.7. On 40 double-scored items, humans and the checklist agree on pass@7 90% of the time, but quadratic-weighted kappa on the 0–10 total is ~0 because the checklist lives in {9,10} while humans use {6,7,8} for wrong-frame drafts. **Pass@9 (0.926) is slightly less inflated; the human study is the actual quality evidence.**

2. **Route accuracy 90.9% is easy to game.** A trivial always-escalate policy already gets 71.4% because the golden set is 71% escalate by design. The number that is not gamed is **false auto (2)** plus **escalate recall (0.988)**.

3. **Copy-nearest beats this agent on cosine-to-Apple (0.119 vs 0.072).** If the assignment were “imitate 2017 Apple Twitter”, the simple baseline wins. Apple’s modal reply is “DM us” + a dead `t.co`. Optimizing that metric produces an agent that *cannot* be auto-trusted.

4. **Intent 78.8% includes language_unsupported at F1=1.0** (script detection is easy) and is helped by keyword overrides on battery/account. Macro-F1 0.773 is the fairer headline; `other` at 0.36 and `software_update_bug` recall 0.56 are the operating reality.

5. **The corpus is iOS 11 / late 2017.** A historically grounded workaround for the I-character bug is correct *for this data* and would be wrong in 2026. Headline metrics do not measure time-shift.

6. **n=231, one brand, one rater** (me). No second annotator, so label noise is unmeasured. Stratified sampling over-represents rare lockouts relative to production volume.

---

## What I would do with one more week

1. **Vision on screenshot tweets** — the two false autos are this. Even a cheap “image present → veto auto” rule is a one-hour fix; a captioner is the real fix.
2. **Thread reconstruction** from `in_response_to_tweet_id` on the original Kaggle CSV, then re-label mid-thread gold.
3. **Multi-label intents + stakes routing** so battery+crash does not depend on keyword order.
4. **Replace template drafts with a small local or API LLM** constrained to: retrieved Apple replies (PII-stripped) + playbook + “do not invent account status”. Re-run the 40-item human study on the new drafts; that is the week’s actual deliverable.
5. **Second annotator on 80 items** for Cohen’s kappa on intent and route. If route kappa < 0.7, the 2-false-auto headline is not yet a product metric.
6. **Time-shift eval:** hold out all 11.1.1-workaround tweets and test whether the agent still auto-sends that article on a 2018-style keyboard complaint. That is the honest test of “grounded in history” vs “memorized a cluster”.

---

## Decision log

1. **Brand = Apple Support.** Highest volume in the dataset, distinct product lines, and a documented public-vs-DM split. A bank brand would have made Banking77 tempting in a way that hides Apple-specific work.
2. **Use the AppleConvos slice, not the 3M CSV.** Same source, 15-minute constraint, no Kaggle credential.
3. **Do not use Banking77.** Domain mismatch; would be decorative.
4. **Gold route ≠ Apple asked for DM.** Inbox-agent policy vs 2017 Twitter culture. Otherwise copy-nearest inherits Apple’s DM habit and looks safe.
5. **Default escalate, auto by allowlist + veto.** False auto is the expensive error. Documented in `apple_agent/escalate.py`.
6. **Hold out every golden ID from retrieval and from classifier training.** Otherwise nearest-neighbor “accuracy” is leakage.
7. **Train intent on weak labels, evaluate on gold.** There is no large human-labelled train set; pretending otherwise would be a silent leak from the 231.
8. **TF-IDF, not embeddings.** Reproducible in 15 minutes on CPU; the simple baseline uses the same encoder so the comparison is fair.
9. **No LLM in the default generator.** Same 15-minute / no-key constraint. Retrieval + playbook is the system; an LLM is a week-two upgrade, not a demo dependency.
10. **LLM-as-judge is a written rubric with two backends.** Default = programmatic checklist so `run_eval` is deterministic. `OPENAI_API_KEY` runs the same prompt. Headline table uses programmatic; the 40-item human study is the agreement evidence.
11. **Single primary intent, even on dual-symptom tweets.** Route uses stakes vetoes so a missed second intent does not auto-send. Labelled the higher-stakes class in gold.
12. **Generalize version pins in drafts** (don’t tell people to install iOS 11.1.1). Historical URLs from retrieved Apple replies are kept when they are articles, not the generic “DM us” shortlink.
13. **Drop 19 candidates** that were thank-yous or fragments (`iPhone 7 iOS 11.1.1`). Scoring those would have been theatre.
14. **Report false auto and pass@9 alongside accuracy.** Accuracy on an escalate-heavy set, and a saturating judge, are both easy to misread — see above.
15. **Stratify the golden draw** so language, lockout, AirPods, and billing exist in n=231. A uniform sample would have been ~30% generic iOS-11 rage and an untestable tail.

---

## Repo map

```
apple_agent/          # intent, retrieve, escalate, draft, judge, metrics
scripts/download_data.py
scripts/build_index.py
scripts/build_golden.py
scripts/run_eval.py
scripts/demo.py
data/golden/          # labels.tsv, golden.jsonl, SAMPLING.md, human_judge.jsonl
data/playbook.yaml    # historical Apple moves, documented
results/metrics.json  # last eval snapshot
```
