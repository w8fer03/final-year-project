"""
identify_language_reviews.py

Identify RAW Shopee reviews as Malay, English, Mixed (Malay-English),
or Review_Required before full preprocessing.
"""

import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

# ==============================
# CONFIGURATION
# ==============================
INPUT_FILE = "dataset v3 labelled 2.csv"  # change if needed
COMMENT_COLUMN = "comment"
STARS_COLUMN = "stars"
EXCLUDE_THREE_STAR = True
REMOVE_DUPLICATE_REVIEW_IDS = True
MALAY_NORMALIZATION_JSON = "malay_normalization_map_v2.json"

POSSIBLE_ID_COLUMNS = [
    ["item_id", "cmtid"],
    ["itemid", "cmtid"],
    ["product_id", "cmtid"],
    ["cmtid"],
]

# ==============================
# LANGUAGE LEXICONS
# ==============================
MALAY_WORDS = {
    "aku", "anda", "awak", "dia", "kami", "kamu", "kita", "mereka", "saya",
    "ini", "itu", "sini", "situ", "sana", "yang", "dan", "atau", "tetapi",
    "tapi", "namun", "kerana", "sebab", "kalau", "jika", "bila", "apabila",
    "dengan", "tanpa", "untuk", "daripada", "dari", "kepada", "pada", "dalam",
    "luar", "atas", "bawah", "dekat", "jauh", "selepas", "sebelum", "sudah",
    "belum", "masih", "akan", "boleh", "dapat", "mesti", "perlu", "mahu", "nak",
    "hendak", "tidak", "tak", "bukan", "jangan", "pun", "juga", "sahaja", "saja",
    "lagi", "memang", "sangat", "terlalu", "lebih", "kurang", "semua", "banyak",
    "sedikit", "setiap", "satu", "dua", "tiga", "empat", "lima", "orang", "apa",
    "kenapa", "mengapa", "bagaimana", "mana", "siapa", "berapa", "ada", "tiada",
    "guna", "pakai", "beli", "jual", "cuba", "terima", "sampai", "hantar", "ambil",
    "beri", "dengar", "lihat", "nampak", "rasa", "suka", "puas", "kecewa", "marah",
    "sedih", "senang", "susah", "cepat", "lambat", "bagus", "baik", "teruk", "rosak",
    "cantik", "murah", "mahal", "jelas", "kuat", "lemah", "lama", "baru", "betul",
    "salah", "berbaloi", "selesa", "ringan", "berat", "kemas", "mudah", "sukar",
    "panas", "sejuk", "pendek", "panjang", "kecil", "besar", "asli", "barang",
    "produk", "harga", "penjual", "kedai", "pesanan", "bungkusan", "pembungkusan",
    "penghantaran", "kurier", "pos", "stok", "waranti", "jaminan", "pemulangan",
    "pulangan", "bayaran", "duit", "wang", "kualiti", "fungsi", "bateri", "bunyi",
    "suara", "muzik", "panggilan", "wayar", "kepala", "telinga", "pengecas", "cas",
    "sambungan", "sambung", "putus", "pecah", "retak", "hilang", "masalah", "lah",
    "je", "jer", "kan", "ke", "kot", "wei", "weh", "bah"
}

MALAY_SLANG = {
    "aq", "ak", "sy", "sye", "sya", "ko", "kau", "korang", "diorang", "ni", "nie",
    "nih", "tu", "tuh", "yg", "dgn", "dengn", "utk", "sbb", "sbab", "mmg", "mcm",
    "cam", "cmne", "mne", "mane", "skrg", "skang", "dah", "dh", "sdh", "blm", "tk",
    "tdk", "x", "xde", "xada", "takde", "tkde", "takda", "xnak", "taknak", "blh",
    "bleh", "buleh", "sngt", "sgt", "brg", "org", "dkt", "smpai", "smpi", "nmpak",
    "btl", "btul", "klau", "klu", "kalo", "dngr", "dgar", "kluar", "pon", "tq",
    "tqsm", "thx", "terbaek", "padu", "mantap", "hauk", "byk", "bnyk", "smua", "lps",
    "pastu", "pastuh", "sbbtu", "kene", "kne", "tmpt", "bru", "jgn", "jd", "jdi",
    "nk", "korg", "kitorg", "kitorang", "xtau", "taktau", "xleh", "takleh", "xpasti",
    "xpuas", "takpuas", "nye", "nyer", "la", "laa", "lahh", "aa"
}

ENGLISH_WORDS = {
    "i", "me", "my", "mine", "you", "your", "yours", "he", "she", "it", "we", "they",
    "this", "that", "these", "those", "a", "an", "the", "and", "or", "but", "because",
    "if", "when", "while", "although", "with", "without", "for", "from", "to", "of", "in",
    "on", "at", "by", "after", "before", "under", "over", "into", "out", "up", "down",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have",
    "has", "had", "can", "could", "will", "would", "should", "may", "might", "must", "not",
    "no", "yes", "very", "really", "too", "so", "more", "less", "much", "many", "some",
    "all", "any", "still", "already", "only", "also", "again", "never", "always", "then",
    "good", "great", "excellent", "amazing", "awesome", "perfect", "nice", "love", "like",
    "liked", "satisfied", "happy", "recommend", "recommended", "worth", "value", "comfortable",
    "clear", "strong", "fast", "cheap", "affordable", "beautiful", "solid", "useful", "bad",
    "poor", "terrible", "worst", "awful", "broken", "faulty", "disappointed", "disappointing",
    "waste", "expensive", "slow", "weak", "unclear", "uncomfortable", "useless", "fake",
    "damaged", "problem", "issue", "issues", "error", "product", "item", "seller", "shop",
    "store", "order", "parcel", "package", "packaging", "delivery", "shipping", "courier",
    "price", "refund", "return", "warranty", "quality", "sound", "audio", "music", "voice",
    "mic", "microphone", "bass", "treble", "battery", "charging", "charge", "charger", "cable",
    "wire", "connection", "connect", "connected", "disconnect", "pair", "pairing", "earphone",
    "earphones", "earbud", "earbuds", "headphone", "headphones", "phone", "case", "button",
    "buttons", "volume", "noise", "cancel", "cancellation", "working", "works", "work", "use",
    "used", "using", "received", "receive", "arrived", "arrive", "bought", "buy", "purchase",
    "replacement", "original", "performance", "function", "functions", "fit", "size", "small",
    "big", "large", "light", "heavy", "long", "short", "day", "days", "week", "weeks",
    "month", "months"
}

ENGLISH_CONTRACTIONS = {
    "can't", "cannot", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't",
    "isn't", "wasn't", "weren't", "won't", "wouldn't", "shouldn't", "mustn't", "i'm", "i've",
    "i'll", "i'd", "you're", "you've", "you'll", "you'd", "he's", "she's", "it's", "we're",
    "we've", "we'll", "they're", "they've", "they'll", "that's", "there's", "here's", "what's"
}

NEUTRAL_DOMAIN_WORDS = {
    "baseus", "shopee", "bluetooth", "usb", "usbc", "typec", "type-c", "tws", "anc", "enc",
    "ios", "android", "iphone", "ipad", "airpods", "samsung", "xiaomi", "huawei", "oppo", "vivo",
    "realme", "windows", "macos", "pc", "nfc", "aac", "sbc", "aptx", "ldac"
}

AMBIGUOUS_WORDS = {
    "ok", "okay", "okayyy", "haha", "hahaha", "hehe", "hehehe", "hihi", "lol", "lmao", "hmm",
    "hmmm", "erm", "em", "ah", "oh", "ooo", "wow", "wah", "huhu", "heheh"
}

SHARED_AMBIGUOUS = {"air", "jam", "die", "gift"}

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)?")
CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


def load_csv_with_fallback(path: str) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            pass
    raise RuntimeError("Could not read CSV with common encodings.")


def load_extra_malay_evidence(json_path: str):
    path = Path(json_path)
    if not path.exists():
        print(f"[INFO] Optional Malay dictionary not found: {json_path}")
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        added = set()
        if isinstance(mapping, dict):
            for slang, formal in mapping.items():
                added.update(TOKEN_RE.findall(str(slang).lower()))
                added.update(TOKEN_RE.findall(str(formal).lower()))
        MALAY_WORDS.update(added)
        MALAY_SLANG.update(added)
        print(f"[INFO] Added {len(added)} Malay evidence tokens from {json_path}")
    except Exception as exc:
        print(f"[WARNING] Could not load {json_path}: {exc}")


def normalize_detection_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = html.unescape(text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def extract_tokens(text: str):
    return [t.lower() for t in TOKEN_RE.findall(text)]


def token_category(token: str) -> str:
    token = token.lower().strip()
    if token in NEUTRAL_DOMAIN_WORDS:
        return "neutral"
    if token in AMBIGUOUS_WORDS or token in SHARED_AMBIGUOUS:
        return "ambiguous"
    if token in ENGLISH_CONTRACTIONS:
        return "english"
    in_ms = token in MALAY_WORDS or token in MALAY_SLANG
    in_en = token in ENGLISH_WORDS
    if in_ms and in_en:
        return "ambiguous"
    if in_ms:
        return "malay"
    if in_en:
        return "english"
    return "unknown"


def identify_language(text: str) -> dict:
    original = "" if pd.isna(text) else str(text)
    detection_text = normalize_detection_text(original)
    tokens = extract_tokens(detection_text)

    malay_hits, english_hits, neutral_hits = [], [], []
    ambiguous_hits, unknown_hits = [], []

    for token in tokens:
        category = token_category(token)
        if category == "malay":
            malay_hits.append(token)
        elif category == "english":
            english_hits.append(token)
        elif category == "neutral":
            neutral_hits.append(token)
        elif category == "ambiguous":
            ambiguous_hits.append(token)
        else:
            unknown_hits.append(token)

    ms = len(malay_hits)
    en = len(english_hits)

    if CJK_RE.search(original):
        language = "Review_Required"
        reason = "contains_non_target_script"
    elif ms > 0 and en > 0:
        language = "Mixed"
        reason = ""
    elif ms > 0:
        language = "Malay"
        reason = ""
    elif en > 0:
        language = "English"
        reason = ""
    else:
        language = "Review_Required"
        reason = "insufficient_language_evidence"

    return {
        "language_detection_text": detection_text,
        "language_type": language,
        "language_review_reason": reason,
        "malay_hit_count": ms,
        "english_hit_count": en,
        "neutral_hit_count": len(neutral_hits),
        "ambiguous_hit_count": len(ambiguous_hits),
        "unknown_hit_count": len(unknown_hits),
        "malay_hits": " | ".join(malay_hits),
        "english_hits": " | ".join(english_hits),
        "neutral_hits": " | ".join(neutral_hits),
        "ambiguous_hits": " | ".join(ambiguous_hits),
        "unknown_hits": " | ".join(unknown_hits),
    }


def find_duplicate_columns(df: pd.DataFrame):
    columns = set(df.columns)
    for candidate in POSSIBLE_ID_COLUMNS:
        if all(col in columns for col in candidate):
            return candidate
    return None


def prepare_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if STARS_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{STARS_COLUMN}' not found. Available: {list(df.columns)}"
        )

    df = df.copy()
    stars = pd.to_numeric(df[STARS_COLUMN], errors="coerce")
    df["_stars_numeric"] = stars
    df["sentiment_label"] = pd.NA
    df.loc[stars.isin([1, 2]), "sentiment_label"] = "Negative"
    df.loc[stars.isin([4, 5]), "sentiment_label"] = "Positive"

    if EXCLUDE_THREE_STAR:
        before = len(df)
        df = df[stars != 3].copy()
        print(f"[INFO] Excluded {before - len(df):,} three-star reviews.")

    return df


def main():
    print("\nStarting raw-review language identification...\n")

    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Put the CSV in the same folder or change INPUT_FILE."
        )

    load_extra_malay_evidence(MALAY_NORMALIZATION_JSON)
    df = load_csv_with_fallback(INPUT_FILE)
    print(f"[INFO] Loaded {len(df):,} rows.")

    if COMMENT_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{COMMENT_COLUMN}' not found. Available: {list(df.columns)}"
        )

    df.insert(0, "source_row_number", range(2, len(df) + 2))

    comments = df[COMMENT_COLUMN].astype("string")
    mask = (
        comments.notna()
        & comments.str.strip().ne("")
        & comments.str.lower().ne("nan")
    )
    before = len(df)
    df = df[mask].copy()
    print(f"[INFO] Removed {before - len(df):,} null/empty reviews.")

    if REMOVE_DUPLICATE_REVIEW_IDS:
        dup_cols = find_duplicate_columns(df)
        if dup_cols:
            before = len(df)
            df = df.drop_duplicates(subset=dup_cols, keep="first").copy()
            print(f"[INFO] Removed {before - len(df):,} duplicate IDs using {dup_cols}.")
        else:
            print("[INFO] No review-ID columns found; no text-based deduplication done.")

    df = prepare_sentiment(df)

    print("[INFO] Identifying Malay / English / Mixed...")
    detected = df[COMMENT_COLUMN].apply(identify_language)
    detected_df = pd.DataFrame(detected.tolist(), index=df.index)
    df = pd.concat([df, detected_df], axis=1)

    # Invalid/missing stars should not go into model datasets.
    invalid_star = ~df["_stars_numeric"].isin([1, 2, 4, 5])
    df.loc[invalid_star, "language_type"] = "Review_Required"
    df.loc[invalid_star, "language_review_reason"] = "invalid_or_missing_star_rating"

    df["language_sentiment_group"] = pd.NA
    ready = (
        df["language_type"].isin(["Malay", "English", "Mixed"])
        & df["sentiment_label"].isin(["Positive", "Negative"])
    )
    df.loc[ready, "language_sentiment_group"] = (
        df.loc[ready, "language_type"] + "_" + df.loc[ready, "sentiment_label"]
    )

    df = df.drop(columns=["_stars_numeric"])

    df_malay = df[df["language_type"] == "Malay"].copy()
    df_english = df[df["language_type"] == "English"].copy()
    df_mixed = df[df["language_type"] == "Mixed"].copy()
    df_review = df[df["language_type"] == "Review_Required"].copy()

    df.to_csv("language_identified_reviews.csv", index=False, encoding="utf-8-sig")
    df_malay.to_csv("dataset_malay.csv", index=False, encoding="utf-8-sig")
    df_english.to_csv("dataset_english.csv", index=False, encoding="utf-8-sig")
    df_mixed.to_csv("dataset_mixed.csv", index=False, encoding="utf-8-sig")
    df_review.to_csv("language_review_required.csv", index=False, encoding="utf-8-sig")

    distribution = (
        df.groupby(["language_type", "sentiment_label"], dropna=False)
          .size()
          .reset_index(name="count")
    )
    distribution.to_csv("language_distribution.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("LANGUAGE COUNTS")
    print("=" * 70)
    print(df["language_type"].value_counts(dropna=False).to_string())

    print("\nLANGUAGE x SENTIMENT")
    print(pd.crosstab(df["language_type"], df["sentiment_label"], margins=True).to_string())

    print("\n[DONE] Created:")
    print("  language_identified_reviews.csv")
    print("  dataset_malay.csv")
    print("  dataset_english.csv")
    print("  dataset_mixed.csv")
    print("  language_review_required.csv")
    print("  language_distribution.csv")
    print("\nIMPORTANT: inspect Review_Required and samples from all 3 classes before training.")


if __name__ == "__main__":
    main()
