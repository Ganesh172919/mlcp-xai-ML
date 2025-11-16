# Dataset Card: data-all-annotations

This folder contains raw text files with IDs and annotations. You will typically parse these into a tabular format (CSV/Parquet) before modeling.

## Contents
- `readme.txt`: Source notes and format hints.
- `trainingdata-*.txt`: Training split annotations and IDs.
- `trialdata-*.txt`: Small trial/dev split.
- `testdata-*.txt`: Test split (often IDs only; labels may be held‑out).
- Files suffixed with `all-annotations` contain multiple annotations per item (e.g., crowd labels). Files suffixed with `ids` list only the identifiers.
- Files mentioning `taskA` / `taskB` indicate different labeling tasks (e.g., binary vs. multi‑label). See the original task description in `readme.txt` for specifics.

## Typical parsing approach
1. Inspect a sample line in each file to confirm separators (space, tab, or CSV‑like).
2. Load and split lines into columns (`id`, `text`, `label`, etc.).
3. If multiple annotations exist for the same ID, aggregate to a single label (e.g., majority vote) or keep annotator‑level rows if you study label variance.
4. Save a clean dataset for modeling, e.g. `data/processed/train.csv` with columns like:
   - `id`, `text`, `label` (binary 0/1 or task‑specific classes)

## Example: quick majority‑vote aggregation in Python
```python
import csv
from collections import defaultdict, Counter

# Adjust these based on the actual file format after inspection
INPUT = 'data-all-annotations/trainingdata-all-annotations.txt'
OUTPUT = 'data/processed/train.csv'

rows_by_id = defaultdict(list)
with open(INPUT, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Example placeholder parsing; update to match the true delimiter/order
        # e.g., id\tlabel\ttext or id\tannotator\tlabel\ttext
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        _id = parts[0]
        label = parts[1]
        text = parts[-1]
        rows_by_id[_id].append((label, text))

parsed = []
for _id, items in rows_by_id.items():
    labels = [l for l, _ in items]
    text = items[0][1]  # assume identical text across annotations
    majority = Counter(labels).most_common(1)[0][0]
    parsed.append((_id, text, majority))

# Ensure output dir exists
import os
os.makedirs('data/processed', exist_ok=True)

with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['id', 'text', 'label'])
    w.writerows(parsed)

print(f'Saved {len(parsed)} rows to {OUTPUT}')
```

## Recommended next steps
- Clarify task definitions (A/B) in `readme.txt` and decide which task to model first.
- Verify class balance after aggregation. If imbalanced (likely), continue with the project’s oversampling workflow.
- Keep raw files read‑only; write processed artifacts to `data/processed/`.
