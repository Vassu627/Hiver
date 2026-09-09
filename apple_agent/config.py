from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_CSV = DATA / "raw" / "apple_support.csv"
CORPUS_PATH = DATA / "processed" / "corpus.jsonl.gz"
INDEX_DIR = DATA / "processed" / "index"
PLAYBOOK_PATH = DATA / "playbook.yaml"
GOLDEN_PATH = DATA / "golden" / "golden.jsonl"
HUMAN_JUDGE_PATH = DATA / "golden" / "human_judge.jsonl"
RESULTS_DIR = ROOT / "results"

BRAND = "Apple Support"
BRAND_ALIASES = ("applesupport", "@115858", "@116333", "@applesupport")

INTENTS = [
    "keyboard_input",
    "battery_power",
    "software_update_bug",
    "connectivity",
    "hardware",
    "icloud_data",
    "billing_purchases",
    "account_security",
    "accessories",
    "macos",
    "how_to_feature",
    "language_unsupported",
    "other",
]

# Auto-send is allowed only when a public, non-account playbook step exists.
AUTO_INTENTS_DEFAULT = {
    "keyboard_input",
    "language_unsupported",
    "how_to_feature",
}
