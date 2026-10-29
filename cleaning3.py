import pandas as pd
import re
import malaya
from collections import Counter

# ==========================================
# Canonicalization map: Malaya's normalizer translates some Malay
# words to English and leaves others as-is, which splits words like
# "good"/"baik" or "quality"/"kualiti" into two separate tokens.
# This map collapses each concept back into ONE canonical English
# token. Only content words (product/sentiment-bearing) are mapped;
# Malay function/grammar words (yang, tu, ni, dan, dengan, pun, la,
# dah...) are left as-is since they carry syntactic, not sentiment,
# information and forcing a translation risks changing meaning.
# ==========================================
CANONICAL_MAP = {
    # good / positive quality
    "baik": "good", "bagus": "good", "mantap": "good", "padu": "good",
    "sedap": "good", "power": "good",
    # best / excellent
    "terbaik": "best", "excellent": "best",
    # quality
    "kualiti": "quality",
    # price / value
    "harga": "price", "murah": "cheap", "berbaloi": "worth it",
    # speed
    "cepat": "fast", "pantas": "fast", "lambat": "slow", "lama": "long",
    # sound
    "bunyi": "sound", "dengar": "hear", "audio": "sound",
    # delivery / shipping
    "penghantaran": "delivery", "hantar": "send",
    # item / product
    "barang": "item", "produk": "product",
    # body parts / fit
    "telinga": "ear", "besar": "big", "kecil": "small",
    # buy / use
    "beli": "buy", "pakai": "use", "guna": "use",
    # condition
    "rosak": "broken", "teruk": "bad", "elok": "fine",
    # comfort
    "selesa": "comfortable",
    # thanks
    "terima": "thank", "kasih": "you",  # "terima kasih" -> "thank you"
    # recommend
    "cadang": "recommend", "disyorkan": "recommended",
    # clarity
    "jelas": "clear",
    # can / cannot
    "boleh": "can",
    # condition
    "keadaan": "condition",

    # ------------------------------------------------------------
    # Informal Malay spelling variants left inconsistent by Malaya's
    # rule-based normalizer (same word, two forms both survive).
    # Mapped to their standard Malay spelling, not translated to
    # English, since these are function/connector words.
    # ------------------------------------------------------------
    "yg": "yang",
    "dgn": "dengan",
    "mmg": "memang",
    "tak": "tidak",
    "la": "lah",
    "sbb": "sebab",
    "nk": "nak",
}

CAN_NOT_RE = re.compile(r"\bcan not\b")


def canonicalize_synonyms(text):
    """e) Merge EN/MY synonym pairs into one canonical token, and fix
    'can not' -> 'cannot' so it matches contraction-expansion intent."""
    if not isinstance(text, str):
        return text
    tokens = text.split()
    tokens = [CANONICAL_MAP.get(tok, tok) for tok in tokens]
    text = " ".join(tokens)
    text = CAN_NOT_RE.sub("cannot", text)
    return text


# 1. Load the rule-based normalizer from Malaya
# We load this outside the function so it doesn't reload for every single row
normalizer = malaya.normalizer.rules.load()

# 2. Load the dataset
df = pd.read_csv("dataset v3 labelled 2.csv")
df['comment'] = df['comment'].astype(str)
df['comment'] = df['comment'].replace('nan', pd.NA)

# ==========================================
# a) Remove null values and duplicated reviews
# ==========================================
df_clean = df.dropna(subset=['comment']).copy()
df_clean = df_clean.drop_duplicates(subset=['comment'], keep='first')

# ==========================================
# b) Delete rows with foreign/non-ASCII characters
# ==========================================
def contains_foreign_chars(text):
    return bool(re.search(r'[^\x00-\x7F]', str(text)))

mask_foreign = df_clean['comment'].apply(contains_foreign_chars)
df_clean = df_clean[~mask_foreign].copy()

# ==========================================
# Text Normalization via Malaya
# ==========================================
def malaya_normalize_text(text):
    if not isinstance(text, str):
        return text
        
    # b) Remove URLs and hyperlinks BEFORE passing to Malaya
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Let Malaya handle slang, elongation, emojis, lowercase, and Malaysian context
    result = normalizer.normalize(text)
    text = result['normalize'] 
    
    #g) Convert all text to lowercase -- NEW: Force the text to lowercase before the strict filter
    text = text.lower()
    
    # c) STRICT FILTER: Remove punctuation and non-Latin characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # d) Remove extra whitespace and newline characters
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply the Malaya normalization function
print("Starting Malaya normalization (this may take a few minutes depending on dataset size)...")
df_clean['cleaned_comment'] = df_clean['comment'].apply(malaya_normalize_text)

# ==========================================
# Drop rows where Malaya normalization produced an empty/null result
# (e.g. punctuation-only comments like ",,.,.,.,.,.")
# ==========================================
before = len(df_clean)
df_clean = df_clean.dropna(subset=['cleaned_comment']).copy()
df_clean = df_clean[df_clean['cleaned_comment'].str.strip() != '']
print(f"Dropped {before - len(df_clean)} rows that became empty after normalization")

vocab_before = Counter(' '.join(df_clean['cleaned_comment']).split())
print(f"Vocab size before reconciliation: {len(vocab_before)} unique tokens")

# ==========================================
# e) Reconcile EN/MY synonym pairs into one canonical token
#    (e.g. "baik"/"bagus"/"mantap" -> "good", "kualiti" -> "quality")
# ==========================================
df_clean['reconciled_comment'] = df_clean['cleaned_comment'].apply(canonicalize_synonyms)

# Re-dedupe: canonicalization can make previously-distinct rows identical
# (e.g. one row written with "baik", another with "good")
before = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['reconciled_comment'], keep='first')
print(f"Dropped {before - len(df_clean)} rows that became duplicates after reconciliation")

vocab_after = Counter(' '.join(df_clean['reconciled_comment']).split())
print(f"Vocab size after reconciliation: {len(vocab_after)} unique tokens")

# Save your fully cleaned dataset to a new CSV file
df_clean.to_csv("dataset_v3_malaya_cleaned3.csv", index=False, encoding='utf-8-sig')
print(f"Data cleaning complete. Final dataset contains {len(df_clean)} rows.")