# Data

- `raw/apple_support.csv` — downloaded by `scripts/download_data.py` from Hugging Face `OpenArchive/AppleConvos` (Apple-filtered [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter), CC BY-NC-SA 4.0). Not committed.
- `processed/` — retrieval corpus + sklearn index, built by `scripts/build_index.py`. Golden IDs held out. Not committed; rebuild takes ~90s.
- `golden/` — the evaluation set. Committed.
- `playbook.yaml` — historical Apple Support moves used as draft policy comments.

Do not ship customer tweets outside this assignment; hashed user ids are already in the public file.
