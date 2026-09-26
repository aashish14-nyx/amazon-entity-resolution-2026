# Amazon ML Challenge 2026 — Business Entity Resolution

End-to-end entity resolution pipeline for matching business records across multiple independent data sources with noisy, multilingual, and inconsistent fields.

## Features

✅ **Country-Agnostic Design** — Dynamically handles any country (US, India, France, or future regions)  
✅ **Multilingual Support** — Unicode-aware normalization; no language assumptions  
✅ **Memory-Bounded EDA** — Streams large files in chunks to keep memory bounded  
✅ **Two-Stage Pipeline** — Blocking for candidate generation + scoring for final matching  
✅ **F_0.5 Optimized** — Conservative threshold favors precision over recall  
✅ **No External Data** — Uses only provided training/test files  
✅ **Strict Output Validation** — Ensures compliance with submission rules  

## Problem Statement

Source 1 is the deduplicated reference source. Given records from Sources 1, 2, and 3:
- **Task:** For each Source 1 entity, find all matching records from Source 2 and Source 3.
- **Challenge:** Records are noisy (typos, abbreviations, missing fields, transliteration variants, address variations).
- **Metric:** F_0.5 score (precision-heavy; false matches are 2× costlier than missed matches).
- **Constraint:** No external APIs, geocoding, or business registries allowed.

## Dataset Structure

```
dataset/
├── train/
│   ├── train_source1.tsv      (deduplicated reference)
│   ├── train_source2.tsv      (noisy records)
│   ├── train_source3.tsv      (noisy records)
│   └── train_ground_truth.tsv (matching labels)
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Columns in source files:
- `entity_id` — Unique ID with prefix S1-, S2-, or S3-
- `business_name` — May contain typos, abbreviations, transliteration variants
- `business_address` — May contain missing components, abbreviations, formatting differences
- `country` — Open set of string labels (train: US, India; test: +France, possibly others)

## Installation

```bash
git clone https://github.com/aashish14-nyx/amazon-entity-resolution-2026.git
cd amazon-entity-resolution-2026
pip install -r requirements.txt
```

## Usage

### 1. Run EDA (Exploratory Data Analysis)

```bash
python src/eda.py --data-dir dataset
```

Output:
- Memory-bounded streaming profile of all source files
- Entity ID duplicate detection
- Country distribution (including unseen test countries)
- Empty-field statistics
- Ground-truth match-count distribution
- Country consistency checks across matched pairs

### 2. Run End-to-End Pipeline

```bash
python src/cli.py \
  --data-dir dataset \
  --output-dir output
```

**Options:**
```
--top-k 15              Number of TF-IDF candidates per field (default: 15)
--threshold 0.72        Matching score threshold (default: 0.72, higher = more precision)
--skip-eda              Skip diagnostic output
```

**Output:**
- `output/matching_results.tsv` — Final matches (uploaded to leaderboard)
- `output/candidate_pairs.tsv` — Candidate set before final scoring

### 3. Validate Output

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Must print `PASS` before submission.

## Pipeline Architecture

### Stage 1: Candidate Generation (Blocking)

**Goal:** Efficiently find plausible matches for each Source 1 entity without scoring every pair.

**Methods:**
1. **Exact Matching** — Same country + exact normalized name or address
2. **TF-IDF Character N-Grams** — Same country + top-k most similar by business_name and business_address

**Why character n-grams?**
- Work on any language (no language assumption)
- Robust to typos, abbreviations, punctuation differences
- Handle transliteration variants naturally
- Support accented text (e.g., French diacritics) and mixed scripts

**Key Property:** Same country by default (blocking rule). Cross-country matching is possible but rare in training data; pipeline uses strong evidence only.

### Stage 2: Matching (Scoring)

**Goal:** Rank candidate pairs and select matches above a conservative threshold.

**Features per pair:**
- Exact normalized name match (0 or 1)
- Exact normalized address match (0 or 1)
- Sequence similarity (name and address)
- Token Jaccard similarity (name and address)
- Country equality flag

**Scoring Formula:**
```
score = 0.55 * max(name_exact, name_sequence, name_jaccard)
       + 0.35 * max(address_exact, address_sequence, address_jaccard)
       + 0.10 * country_match
```

**Special Case:** If name and address are exact matches and countries match, score = 1.0 (very high confidence).

**Threshold:** Default 0.72 (conservative). Increase for higher precision, decrease for higher recall.

## Multilingual Design

### Unicode Normalization

All text is normalized using `unicodedata.normalize('NFKC')` + casefold + punctuation-to-space replacement:

```python
def normalize_unicode(value: str) -> str:
    value = unicodedata.normalize('NFKC', value)
    value = value.casefold()
    value = ''.join(char if char.isalnum() else ' ' for char in value)
    return re.sub(r'\s+', ' ', value).strip()
```

**Does NOT:**
- Remove non-Latin scripts (Hindi, Arabic, Cyrillic, etc.)
- Assume any particular language
- Translate or transliterate (unless doing local, language-agnostic character n-grams)

**Result:** Works for:
- English abbreviations (Corp → corp)
- Hindi (Devanagari) mixed with Latin (transliterated names)
- French accents (Café → cafe)
- Mixed scripts in same address

### Open-Set Country Handling

The pipeline does NOT hard-code countries:

```python
# ✓ Good: Read countries dynamically
for country in df['country'].unique():
    # block and match within this country

# ✗ Bad: Hard-coded check
if country not in ['US', 'India', 'France']:
    raise ValueError("Unknown country")
```

France is handled automatically using the same Unicode normalization and character n-gram features. If the test set includes a fourth country, the pipeline requires zero code changes.

## Output Format

### matching_results.tsv

Tab-separated file with two columns:

```
source1_entity_id    matched_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812
S1-00002             S3-00004
S1-00003
```

**Rules:**
- Exactly one row per Source 1 entity (including singletons)
- `matched_entity_ids` is empty (not NaN) for entities with no matches
- IDs are comma-separated with no spaces
- Only S2- and S3- IDs (no self-matches to S1)
- No duplicate IDs within a list
- Tab-separated, no quoting

### candidate_pairs.tsv

Same format, but contains the full candidate set fed to the matching model:

```
source1_entity_id    candidate_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812,S3-00999
S1-00002             S3-00004
S1-00003
```

**Property:** Every final match must appear in the candidate list. If not, the validator reports a pipeline bug.

## Evaluation Metric: F_0.5

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

**Macro-averaged:** Computed per Source 1 entity, then averaged across all entities.

**Why precision-heavy?**
- False merge (matching two different businesses) is more costly than a missed link
- In real systems, merging distinct businesses breaks downstream logic
- F_0.5 penalizes false positives 2× more than false negatives

**Example:**
```
True matches:     [S2-00047, S3-00812]
Predicted:        [S2-00047, S2-00193, S3-00812]
Precision = 2/3   (2 correct out of 3 predicted)
Recall = 2/2 = 1.0
F_0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = 0.714
```

## Tuning for Leaderboard

### Increase Precision (at cost of recall):

```bash
python src/cli.py --data-dir dataset --output-dir output --threshold 0.80 --top-k 10
```

Effect:
- Higher threshold → fewer matches (higher precision)
- Smaller top-k → fewer candidates considered (faster, tighter blocking)

### Increase Recall (at cost of precision):

```bash
python src/cli.py --data-dir dataset --output-dir output --threshold 0.60 --top-k 25
```

Effect:
- Lower threshold → more matches (higher recall, but more false positives)
- Larger top-k → more candidates (slower, more opportunity to find matches)

## Files

```
src/
├── __init__.py           Package marker
├── io.py                 File I/O and validation
├── normalize.py          Unicode normalization and similarity metrics
├── eda.py                Exploratory data analysis (memory-bounded)
├── blocking.py           Candidate generation
├── features.py           Feature extraction for matching
├── matching.py           Scoring and final matching
├── validate.py           Output validation against submission rules
└── cli.py                Command-line interface

utils/
└── validate_submission.py Official validator (provided by challenge)

output/
├── matching_results.tsv  Final matches for leaderboard
└── candidate_pairs.tsv   Candidate set (for methodology review)
```

## Fair Play

✓ **Allowed:**
- Only provided train/test data
- String similarity, TF-IDF, local transliteration
- Any scikit-learn classifier
- Custom feature engineering
- Hyperparameter tuning

✗ **Prohibited:**
- External APIs (Google Maps, OpenCage, commercial entity resolution services)
- Business registries or government databases
- Geocoding services
- Translation or transliteration from external services
- Web scraping

## Submission Checklist

- [ ] Run `python src/cli.py --data-dir dataset --output-dir output`
- [ ] Run `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` and get `PASS`
- [ ] Review matching_results.tsv: one row per Source 1 entity, all IDs exist
- [ ] Upload matching_results.tsv to leaderboard
- [ ] Prepare final submission package with code/, output/, and documentation

## References

- [Problem statement](https://www.kaggle.com/competitions/amazon-ml-challenge-2026)
- [Unicode normalization](https://unicode.org/reports/tr15/)
- [F_0.5 score](https://en.wikipedia.org/wiki/F-score)
- [TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)

## License

Apache 2.0
