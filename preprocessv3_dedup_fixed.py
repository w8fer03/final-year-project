"""
Preprocessing v3 for Malay, English and Manglish Shopee reviews.

Main corrections compared with preprocessv1.py:
- recomputes text from the ORIGINAL `comment` column, never from an already-cleaned column;
- deduplicates by matching exact source records, NOT unreliable cmtid alone;
- removes the full-sentence Malaya normalizer because it corrupts English/code-mixed text;
- expands English contractions before punctuation handling;
- uses a curated emoji map and neutral semantic wording for ambiguous emojis;
- applies explicit Malay slang/abbreviation mappings only;
- preserves sentiment punctuation (!, ?) and product numbers;
- preserves case by default for distilbert-base-multilingual-cased;
- detects non-target-script, gibberish and off-topic candidates BEFORE destructive cleaning;
- exports flagged rows and a vocabulary audit instead of silently deleting uncertain cases;
- keeps star-rating labels unchanged and only FLAGS possible rating/text conflicts.

Usage in VS Code:
    python preprocessv3_dedup_fixed.py --input "dataset v3 labelled 2.csv"

Use ORIGINAL `comment`, not existing `cleaned_comment`.
If language_type has already been assigned, the column is carried through
unchanged for the single combined model and later subgroup analysis.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"\b[^\W_]+(?:['’][^\W_]+)?\b", flags=re.UNICODE)

# Emoji modifiers are removed before lookup so 👍🏻, 👍🏽, etc. become 👍.
SKIN_TONE_RE = re.compile("[\U0001F3FB-\U0001F3FF]")
VARIATION_SELECTOR_RE = re.compile("[\uFE0E\uFE0F]")
# Broad emoji/symbol blocks used only AFTER known emojis have been translated.
REMAINING_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\u2600-\u27BF"
    "]"
)

# Contractions are expanded before any punctuation cleanup. We deliberately do NOT
# expand generic possessive 's because seller's / product's are not "seller is".
CONTRACTIONS = {
    "can't": "cannot",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "mustn't": "must not",
    "needn't": "need not",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "what's": "what is",
    "who's": "who is",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "we'd": "we would",
    "they'd": "they would"
}

# Malay reduplication commonly appears as word2 in informal text. Restrict this
# conversion to an explicit list so product models such as A2 / X2 are untouched.
REDUPLICATION_BASES = {
    "betul", "baik", "lama", "sama", "barang", "orang", "cepat", "elok",
    "kadang", "jalan", "main", "cuba", "kali"
}

# Ambiguous items are NOT automatically normalized. They are surfaced in the audit.
# Examples: ckp can mean cakap or cukup; x can mean tidak or a model/letter.
AMBIGUOUS_TOKENS = {"ckp", "x", "kt"}

DOMAIN_TERMS = {
    "earphone", "earphones", "earbud", "earbuds", "headphone", "headphones",
    "headset", "audio", "sound", "bunyi", "suara", "mic", "microphone", "mikrofon",
    "bass", "volume", "speaker", "bluetooth", "pair", "pairing", "connect", "connection",
    "cable", "kabel", "jack", "usb", "type", "iphone", "android", "phone", "telefon",
    "device", "peranti", "music", "muzik", "call", "panggilan", "gaming", "game",
    "quality", "kualiti", "performance", "prestasi", "product", "produk", "item", "barang",
    "seller", "penjual", "delivery", "penghantaran", "shipping", "parcel", "package",
    "order", "purchase", "buy", "beli", "received", "receive", "harga", "price",
    "refund", "return", "warranty", "waranti", "rosak", "working", "work", "function",
    "fungsi", "charge", "charging", "battery", "bateri", "telinga", "fit", "comfortable"
}

# Strong hints for obvious copied sports/news material. They are used only as a flag,
# never as the sole automatic reason for deletion.
OFFTOPIC_RED_FLAGS = {
    "lakers", "espn", "season", "playoff", "player", "coach", "league", "contract",
    "veteran", "basketball", "football", "goal", "match", "tournament"
}

POSITIVE_TERMS = {
    "good", "great", "best", "excellent", "nice", "love", "satisfied", "recommended",
    "recommend", "bagus", "baik", "hebat", "puas", "suka", "mantap", "padu"
}
NEGATIVE_TERMS = {
    "bad", "poor", "worst", "useless", "broken", "spoiled", "regret", "terrible",
    "not", "cannot", "teruk", "buruk", "rosak", "kecewa", "marah", "tidak", "tak"
}

ALLOWED_PUNCT = set("!?.,':;/%()-+")

# Protect exact uppercase product and organization abbreviations. Do not skip
# all uppercase slang: TQ/BRG/YG/DGN should still be expanded correctly.
PROTECTED_UPPERCASE_TOKENS = {
    "LG", "TV", "PC", "USB", "ANC", "ENC", "TWS", "RM", "MYR",
    "GPS", "AI", "HD", "HQ", "IOS", "NFC", "LED", "LCD",
}



def load_json_map(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object/dictionary.")
    return {str(k): str(v) for k, v in data.items()}


def normalize_unicode(text: str) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    return (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("–", "-")
            .replace("—", "-")
    )


def expand_contractions(text: str) -> str:
    # Longer contractions first prevents partial matches.
    keys = sorted(CONTRACTIONS, key=len, reverse=True)
    pat = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)", re.IGNORECASE)

    def repl(match: re.Match) -> str:
        src = match.group(0)
        replacement = CONTRACTIONS[src.lower()]
        if src and src[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        return replacement

    return pat.sub(repl, text)


def canonical_emoji(text: str) -> str:
    """Canonical form of emoji for dictionary matching and audit."""
    return VARIATION_SELECTOR_RE.sub("", SKIN_TONE_RE.sub("", text))


def find_unmapped_emojis(text: str, emoji_map: Dict[str, str]) -> List[str]:
    """Audit Unicode emoji not in the curated mapping; don't invent labels."""
    raw = canonical_emoji(normalize_unicode(str(text)))
    remaining = raw
    # Remove known emojis first; longer emoji sequences before shorter ones.
    for emoji_char in sorted((canonical_emoji(x) for x in emoji_map), key=len, reverse=True):
        remaining = remaining.replace(emoji_char, " ")
    return sorted(set(REMAINING_EMOJI_RE.findall(remaining)))


def translate_emojis(text: str, emoji_map: Dict[str, str]) -> str:
    text = canonical_emoji(text)
    normalized_map = {canonical_emoji(k): v for k, v in emoji_map.items()}

    # Translate only curated, known emoji to short semantic phrases. Empty map
    # entries deliberately discard decorative emoji without adding fake opinion.
    for emoji_char in sorted(normalized_map, key=len, reverse=True):
        text = text.replace(emoji_char, f" {normalized_map[emoji_char]} ")

    # Previously unknown emoji were turned into `emoji_lain`, polluting >1000
    # rows. Unknown emoji are now logged in unmapped_emoji_tokens, NOT injected
    # as fake textual sentiment. Do not infer polarity from stars.
    text = REMAINING_EMOJI_RE.sub(" ", text)
    return text


def normalize_elongation(text: str) -> str:
    # Only alphabetic runs are changed; digits/product model numbers are untouched.
    # Consonant runs of 3+ -> one char: besttt -> best, lambattt -> lambat.
    text = re.sub(r"([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])\1{2,}", r"\1", text)
    # Vowel runs of 3+ -> two chars to avoid destroying legitimate English double vowels:
    # cooool -> cool rather than col.
    text = re.sub(r"([aeiouAEIOU])\1{2,}", r"\1\1", text)
    return text


def normalize_reduplication(text: str) -> str:
    # Example: btul2 is handled by the slang map first; betul2 -> betul betul here.
    for base in sorted(REDUPLICATION_BASES, key=len, reverse=True):
        text = re.sub(
            rf"(?<!\w){re.escape(base)}2(?!\w)",
            f"{base} {base}",
            text,
            flags=re.IGNORECASE,
        )
    return text


def compile_normalization_pattern(norm_map: Dict[str, str]) -> re.Pattern:
    keys = sorted(norm_map, key=lambda k: (len(k.split()), len(k)), reverse=True)
    if not keys:
        return re.compile(r"(?!x)x")
    return re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)", flags=re.IGNORECASE)


def apply_normalization_map(text: str, norm_map: Dict[str, str], pattern: re.Pattern) -> str:
    """Apply safe explicit slang/abbreviation normalization.

    Case-insensitive exact word matching for colloquial Malay.
    Protect explicitly known uppercase product abbreviations (especially LG)
    rather than skipping ALL short ALL-CAPS slang such as TQ, BRG and YG.
    """
    def repl(match: re.Match) -> str:
        src = match.group(0)
        if src in PROTECTED_UPPERCASE_TOKENS:
            return src
        replacement = norm_map.get(src.lower())
        return replacement if replacement is not None else src

    return pattern.sub(repl, text)


@lru_cache(maxsize=4096)
def is_latin_letter(ch: str) -> bool:
    if not ch.isalpha():
        return False
    try:
        return "LATIN" in unicodedata.name(ch)
    except ValueError:
        return False


def non_target_script_flag(text: str) -> Tuple[bool, float, int]:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False, 0.0, 0
    non_latin = [ch for ch in letters if not is_latin_letter(ch)]
    ratio = len(non_latin) / len(letters)
    # Conservative: at least 6 non-Latin letters AND >30% of alphabetic content.
    return (len(non_latin) >= 6 and ratio > 0.30), ratio, len(non_latin)


def strip_non_latin_letters(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalpha() and not is_latin_letter(ch):
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def clean_punctuation(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch.isspace() or ch == "_" or ch in ALLOWED_PUNCT:
            out.append(ch)
        else:
            # Replace discarded symbols with whitespace so words do not concatenate.
            out.append(" ")
    return "".join(out)


def words(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def gibberish_flag(text: str) -> bool:
    toks = [t for t in words(text) if t.isalpha()]
    if not toks:
        # Emoji-only text is valid after emoji translation; empty is handled later.
        return False

    vowels = set("aeiou")

    def vowel_ratio(token: str) -> float:
        letters = [c for c in token.lower() if c.isalpha()]
        if not letters:
            return 1.0
        return sum(c in vowels for c in letters) / len(letters)

    # High-confidence random-string case: one long word with almost no vowels.
    if len(toks) == 1 and len(toks[0]) >= 14 and vowel_ratio(toks[0]) < 0.22:
        return True

    joined = "".join(toks)
    if len(joined) >= 24:
        overall_ratio = sum(c in vowels for c in joined.lower()) / len(joined)
        if overall_ratio < 0.15:
            return True
    return False


def off_topic_flag(text: str) -> bool:
    toks = words(text)
    if len(toks) < 35:
        return False
    tokset = set(toks)
    domain_hits = len(tokset & DOMAIN_TERMS)
    red_hits = len(tokset & OFFTOPIC_RED_FLAGS)

    # Long paragraph with no product/e-commerce anchor, or a strong sports/news pattern.
    if len(toks) >= 50 and domain_hits == 0:
        return True
    if red_hits >= 2 and domain_hits == 0:
        return True
    return False


def label_conflict_flag(stars, cleaned_text: str) -> bool:
    """Conservative QA flag only; NEVER changes the star-derived label."""
    try:
        stars = int(stars)
    except Exception:
        return False

    toks = words(cleaned_text)
    if not toks or len(toks) > 10:
        return False
    pos = sum(t in POSITIVE_TERMS for t in toks)
    neg = sum(t in NEGATIVE_TERMS for t in toks)

    if stars in (1, 2) and pos > 0 and neg == 0:
        return True
    if stars in (4, 5) and neg > 0 and pos == 0:
        return True
    return False


def ambiguous_tokens_present(cleaned_text: str) -> str:
    toks = set(words(cleaned_text))
    found = sorted(toks & AMBIGUOUS_TOKENS)
    return ";".join(found)


def preprocess_text(text: str, emoji_map: Dict[str, str], norm_map: Dict[str, str], norm_pattern: re.Pattern, preserve_case: bool) -> str:
    text = normalize_unicode(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    # Do not inject English `and` into Malay-only reviews. Preserve `&` as
    # a separator (rather than choosing a language for the customer).
    text = text.replace("&", " ")
    text = expand_contractions(text)
    text = translate_emojis(text, emoji_map)
    text = normalize_elongation(text)
    text = apply_normalization_map(text, norm_map, norm_pattern)
    text = normalize_reduplication(text)

    # Non-target-script rows have already been flagged. Any isolated leftover
    # non-Latin letters in an otherwise target-language review are treated as noise.
    text = strip_non_latin_letters(text)

    # Keep sentiment-bearing punctuation instead of [^a-z0-9\s].
    text = clean_punctuation(text)
    text = WS_RE.sub(" ", text).strip()

    if not preserve_case:
        text = text.lower()
    return text


def add_reason(existing: str, reason: str) -> str:
    if not existing:
        return reason
    existing_parts = set(existing.split(";"))
    if reason not in existing_parts:
        return existing + ";" + reason
    return existing


def build_vocab_audit(clean_series: Iterable[str]) -> pd.DataFrame:
    counter = Counter()
    for text in clean_series:
        counter.update(words(str(text)))

    rows = []
    for token, count in counter.most_common():
        no_vowel = not any(v in token for v in "aeiou")
        suspicious = (
            token in AMBIGUOUS_TOKENS
            or (2 <= len(token) <= 4 and no_vowel and token not in {"rm", "usb", "mic", "pro"})
            or token == "emoji_lain"
        )
        rows.append({"token": token, "count": count, "suspicious_for_manual_review": suspicious})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset v3 labelled 2.csv")
    parser.add_argument("--emoji-map", default="emoji_map_v3.json")
    parser.add_argument("--normalization-map", default="malay_normalization_map_v3.json")
    parser.add_argument("--prefix", default="dataset_preprocessed_v3")
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase the final text. Leave OFF for distilbert-base-multilingual-cased.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    # Look for dictionaries next to the script if running from another folder.
    script_dir = Path(__file__).resolve().parent
    emoji_path = Path(args.emoji_map)
    norm_path = Path(args.normalization_map)
    if not emoji_path.exists():
        emoji_path = script_dir / args.emoji_map
    if not norm_path.exists():
        norm_path = script_dir / args.normalization_map
    emoji_map = load_json_map(emoji_path)
    norm_map = load_json_map(norm_path)
    norm_pattern = compile_normalization_pattern(norm_map)

    df = pd.read_csv(input_path)
    if "comment" not in df.columns:
        raise ValueError("Input CSV must contain a 'comment' column.")

    df = df.copy()
    df["preprocess_status"] = "keep"
    df["exclusion_reason"] = ""

    # Label from stars exactly as described in the thesis. 3-star reviews are excluded.
    # Pandas 3.x can infer an existing `sentiment` column as StringDtype. Assigning
    # integer 0/1 into that string column raises:
    # TypeError: Invalid value '0' for dtype 'str'.
    # Preserve any existing text label for auditing, then create a fresh nullable
    # integer sentiment column for model training.
    if "stars" in df.columns:
        stars_num = pd.to_numeric(df["stars"], errors="coerce")
        neutral_mask = stars_num.eq(3)
        invalid_star_mask = ~stars_num.isin([1, 2, 3, 4, 5])
        df.loc[neutral_mask, "preprocess_status"] = "flagged"
        df.loc[neutral_mask, "exclusion_reason"] = "neutral_3_star"
        df.loc[invalid_star_mask, "preprocess_status"] = "flagged"
        df.loc[invalid_star_mask, "exclusion_reason"] = "invalid_star_rating"

        # Keep the original sentiment column, if present, so nothing is lost.
        if "sentiment" in df.columns and "sentiment_original" not in df.columns:
            df.rename(columns={"sentiment": "sentiment_original"}, inplace=True)

        # Nullable Int64 is intentional: neutral/invalid rows can remain <NA>.
        df["sentiment"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df.loc[stars_num.isin([1, 2]), "sentiment"] = 0
        df.loc[stars_num.isin([4, 5]), "sentiment"] = 1

    # OPTIONAL language metadata: keep one combined dataset for binary training.
    # If a previous raw-language identification stage provided a language_type
    # column, retain it and create consistent groups for later analysis.
    # Unresolved rows must be reviewed rather than silently trained.
    if "language_type" in df.columns:
        normalized_language = (
            df["language_type"].astype("string").str.strip().str.casefold()
            .map({"malay": "Malay", "english": "English", "mixed": "Manglish", "manglish": "Manglish"})
        )
        df["language_group"] = normalized_language
        unresolved_language_mask = normalized_language.isna() & df["preprocess_status"].eq("keep")
        df.loc[unresolved_language_mask, "preprocess_status"] = "flagged"
        df.loc[unresolved_language_mask, "exclusion_reason"] = "unresolved_language_type"

    # Null/empty comments.
    null_mask = df["comment"].isna() | df["comment"].astype(str).str.strip().eq("")
    df.loc[null_mask, "preprocess_status"] = "flagged"
    df.loc[null_mask, "exclusion_reason"] = "null_or_empty_comment"

    # DEDUPLICATION FIX: some scraped cmtid values are reused across hundreds of
    # DIFFERENT comments (e.g. a placeholder ID). Treating item_id+cmtid alone
    # as the review identity would discard genuine, unique reviews.
    #
    # Safest conservative rule: flag only identical source records, using all
    # available identity/content attributes. In particular, distinguish reviews
    # with the same stored cmtid but different comments or timestamps.
    exact_record_keys = [
        c for c in ["shop_id", "item_id", "cmtid", "stars", "datetime", "comment"]
        if c in df.columns
    ]
    active_indices = df.index[df["preprocess_status"].eq("keep")]
    if exact_record_keys:
        duplicate_within_active = df.loc[active_indices].duplicated(
            subset=exact_record_keys, keep="first"
        )
        duplicate_indices = active_indices[duplicate_within_active.to_numpy()]
        df.loc[duplicate_indices, "preprocess_status"] = "flagged"
        df.loc[duplicate_indices, "exclusion_reason"] = "duplicate_exact_record"

    # Separate audit for recycled/conflicting IDs. These are NOT automatically
    # duplicates; their source texts are different and should be retained unless
    # another quality rule or manual review provides a reason to exclude them.
    id_collision_audit = pd.DataFrame()
    if "cmtid" in df.columns:
        id_keys = [c for c in ["item_id", "cmtid"] if c in df.columns]
        source_cols = [c for c in ["stars", "datetime", "comment"] if c in df.columns]
        if source_cols:
            has_id = df[id_keys].notna().all(axis=1)
            candidates = df.loc[has_id, id_keys + source_cols].copy()
            distinct_sources = candidates.drop_duplicates(subset=id_keys + source_cols)
            id_sizes = (
                distinct_sources.groupby(id_keys, dropna=False)
                .size().rename("distinct_records_sharing_id")
                .reset_index()
            )
            reused = id_sizes.loc[id_sizes["distinct_records_sharing_id"] > 1]
            if len(reused):
                id_collision_audit = distinct_sources.merge(reused, on=id_keys)

    cleaned_values = []
    nonlatin_ratios = []
    nonlatin_counts = []
    label_conflicts = []
    ambiguous_lists = []
    unmapped_emoji_lists = []

    for idx, row in df.iterrows():
        raw = "" if pd.isna(row.get("comment")) else str(row.get("comment"))
        raw_norm = normalize_unicode(raw)
        unmapped_emoji_lists.append(" | ".join(find_unmapped_emojis(raw, emoji_map)))
        is_non_target, ratio, count_nonlatin = non_target_script_flag(raw_norm)
        nonlatin_ratios.append(round(ratio, 4))
        nonlatin_counts.append(count_nonlatin)

        # Build cleaned text even for flagged rows so the audit is inspectable.
        cleaned = preprocess_text(raw, emoji_map, norm_map, norm_pattern, preserve_case=not args.lowercase) if raw else ""
        cleaned_values.append(cleaned)

        if df.at[idx, "preprocess_status"] == "keep":
            reasons = ""
            if is_non_target:
                reasons = add_reason(reasons, "predominantly_non_target_script")

            # Gibberish is evaluated after URL/emoji normalization but before using the row.
            gib_text = translate_emojis(URL_RE.sub(" ", raw_norm), emoji_map)
            if gibberish_flag(gib_text):
                reasons = add_reason(reasons, "gibberish_candidate")

            if off_topic_flag(raw_norm):
                reasons = add_reason(reasons, "off_topic_candidate")

            if not cleaned.strip():
                reasons = add_reason(reasons, "empty_after_cleaning")

            if reasons:
                df.at[idx, "preprocess_status"] = "flagged"
                df.at[idx, "exclusion_reason"] = reasons

        label_conflicts.append(label_conflict_flag(row.get("stars"), cleaned))
        ambiguous_lists.append(ambiguous_tokens_present(cleaned))

    df["cleaned_comment_v3"] = cleaned_values
    df["unmapped_emoji_tokens"] = unmapped_emoji_lists
    df["non_latin_letter_ratio"] = nonlatin_ratios
    df["non_latin_letter_count"] = nonlatin_counts
    df["label_conflict_candidate"] = label_conflicts
    df["ambiguous_tokens_to_review"] = ambiguous_lists

    # IMPORTANT: no post-cleaning dedupe on cleaned_comment_v2. Two real customers can
    # legitimately write the same short review (e.g., "good" or "bagus").
    clean_df = df[df["preprocess_status"].eq("keep")].copy()
    flagged_df = df[~df["preprocess_status"].eq("keep")].copy()

    # Keep a stable `cleaned_comment` name for downstream model code.
    clean_df["cleaned_comment"] = clean_df["cleaned_comment_v3"]
    flagged_df["cleaned_comment"] = flagged_df["cleaned_comment_v3"]

    clean_path = f"{args.prefix}_clean.csv"
    flagged_path = f"{args.prefix}_flagged.csv"
    vocab_path = f"{args.prefix}_vocabulary_audit.csv"
    summary_path = f"{args.prefix}_audit_summary.json"
    emoji_audit_path = f"{args.prefix}_unmapped_emoji_review.csv"
    id_collision_path = f"{args.prefix}_id_collision_audit.csv"

    clean_df.to_csv(clean_path, index=False, encoding="utf-8-sig")
    flagged_df.to_csv(flagged_path, index=False, encoding="utf-8-sig")
    id_collision_audit.to_csv(id_collision_path, index=False, encoding="utf-8-sig")
    build_vocab_audit(clean_df["cleaned_comment"]).to_csv(vocab_path, index=False, encoding="utf-8-sig")
    emoji_to_review = df[df["unmapped_emoji_tokens"].astype(bool)].copy()
    emoji_to_review[["comment", "unmapped_emoji_tokens", "cleaned_comment_v3", "preprocess_status"]].to_csv(
        emoji_audit_path, index=False, encoding="utf-8-sig"
    )

    summary = {
        "input_rows": int(len(df)),
        "distinct_records_with_reused_cmtid": int(len(id_collision_audit)),
        "reused_cmtid_groups": int(len(id_collision_audit.groupby([c for c in ["item_id", "cmtid"] if c in id_collision_audit.columns]))) if len(id_collision_audit) else 0,
        "kept_rows": int(len(clean_df)),
        "flagged_rows": int(len(flagged_df)),
        "preserve_case": bool(not args.lowercase),
        "flag_counts": Counter(
            reason
            for value in flagged_df["exclusion_reason"].fillna("")
            for reason in str(value).split(";")
            if reason
        ),
        "label_conflict_candidates_kept": int(clean_df["label_conflict_candidate"].sum()),
        "rows_with_ambiguous_tokens_kept": int(clean_df["ambiguous_tokens_to_review"].astype(bool).sum()),
        "rows_with_unmapped_emojis": int(len(emoji_to_review)),
        "language_type_preserved": bool("language_type" in df.columns),
        "language_group_counts_kept": clean_df["language_group"].value_counts().to_dict() if "language_group" in clean_df.columns else {},
    }
    summary["flag_counts"] = dict(summary["flag_counts"])
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Clean dataset: {clean_path}")
    print(f"Flagged rows: {flagged_path}")
    print(f"Vocabulary audit: {vocab_path}")
    print(f"Audit summary: {summary_path}")
    print(f"Unmapped emoji audit: {emoji_audit_path}")
    print(f"Reused ID audit (not auto-excluded): {id_collision_path}")


if __name__ == "__main__":
    main()
