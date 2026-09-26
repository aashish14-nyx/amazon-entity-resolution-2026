"""Unicode-aware normalization and similarity metrics.

Designed to work with any language, script, and country without
assumptions or hard-coded language-specific rules.
"""

import re
import unicodedata
from difflib import SequenceMatcher


def normalize_unicode(value: str) -> str:
    """Normalize text without assuming any particular language.

    Steps:
      1. NFKC normalization (canonical + compatibility decomposition)
      2. Case-fold (language-aware lowercase)
      3. Replace non-alphanumeric with spaces (preserves script)
      4. Collapse multiple spaces to one

    Does NOT:
      - Remove non-Latin scripts (Devanagari, Arabic, Cyrillic, etc.)
      - Translate or transliterate using external services
      - Strip accents (e.g., é remains in Unicode form)

    Examples:
      - "Café" → "café" (via casefold)
      - "ABC & Corp." → "abc corp"
      - "मेडिकल" (Hindi) → "मेडिकल" (preserved)
      - "ул. Ленина" (Russian) → "ул ленина"
    """
    value = "" if value is None else str(value)
    value = unicodedata.normalize("NFKC", value)
    value = value.casefold()
    value = "".join(char if char.isalnum() else " " for char in value)
    return re.sub(r"\s+", " ", value).strip()


def compact_text(value: str) -> str:
    """Remove all whitespace and punctuation for exact-match comparisons."""
    return re.sub(r"\W+", "", normalize_unicode(value), flags=re.UNICODE)


def tokens(value: str) -> set[str]:
    """Split normalized text into tokens (words)."""
    return set(normalize_unicode(value).split())


def token_jaccard(left: str, right: str) -> float:
    """Jaccard similarity between token sets.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Range: [0, 1]
    - 0: no common tokens
    - 1: identical token sets

    Robust to:
      - word order ("New York" vs "York New")
      - extra words ("New York City" vs "New York")
      - abbreviations if tokens differ ("Corp" vs "Corporation")

    Not robust to:
      - typos within a word ("Corop" vs "Corp")
      - transliteration ("Pvt" vs "Private")
    """
    a = tokens(left)
    b = tokens(right)

    if not a and not b:
        return 0.0

    return len(a & b) / max(1, len(a | b))


def sequence_similarity(left: str, right: str) -> float:
    """Longest common subsequence ratio (via difflib.SequenceMatcher).

    Range: [0, 1]
    - 0: completely different strings
    - 1: identical strings

    Robust to:
      - typos and spelling mistakes
      - missing characters
      - extra characters
      - transposition (partial)

    Example:
      - "Organization" vs "Organisation" ≈ 0.95
      - "Ltd." vs "Limited" ≈ 0.42
    """
    left = normalize_unicode(left)
    right = normalize_unicode(right)

    if not left or not right:
        return 0.0

    return SequenceMatcher(None, left, right).ratio()


def exact_similarity(left: str, right: str) -> float:
    """Binary: 1.0 if normalized and compacted strings are identical, else 0.0.

    Removes all punctuation and whitespace before comparison.

    Example:
      - "ABC Corp." vs "ABC Corp" → 1.0
      - "Ltd." vs "Limited" → 0.0
    """
    left = compact_text(left)
    right = compact_text(right)

    if not left or not right:
        return 0.0

    return 1.0 if left == right else 0.0


def char_ngram_similarity(left: str, right: str, n: int = 3) -> float:
    """Jaccard similarity between character n-grams.

    Useful for:
      - typos and spelling variations
      - transliteration differences
      - abbreviations
      - languages without word boundaries (e.g., CJK)

    Args:
        left: First string
        right: Second string
        n: N-gram size (default 3 for trigrams)

    Returns:
        Jaccard similarity in [0, 1]
    """
    def get_ngrams(s, n):
        s = normalize_unicode(s)
        if len(s) < n:
            return {s}
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    left_grams = get_ngrams(left, n)
    right_grams = get_ngrams(right, n)

    if not left_grams or not right_grams:
        return 0.0

    return len(left_grams & right_grams) / max(1, len(left_grams | right_grams))
