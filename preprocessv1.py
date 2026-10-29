import pandas as pd
import re
import malaya
import json
from collections import Counter

# ==============================================================================
# Load External Dictionaries
# ==============================================================================
with open("emoji_map.json", "r", encoding="utf-8") as f:
    EMOJI_TO_MALAY = json.load(f)

with open("formal_malay_map.json", "r", encoding="utf-8") as f:
    FORMAL_MALAY_MAP = json.load(f)

# 1. Load Malaya's rule-based normalizer
normalizer = malaya.normalizer.rules.load()

# 2. Load dataset
df = pd.read_csv("dataset v3 labelled 2.csv")
df['comment'] = df['comment'].astype(str)
df['comment'] = df['comment'].replace('nan', pd.NA)

# a) Remove null values and duplicated reviews
df_clean = df.dropna(subset=['comment']).copy()
df_clean = df_clean.drop_duplicates(subset=['comment'], keep='first')

# ==============================================================================
# TEXT PREPROCESSING PIPELINE
# ==============================================================================
def preprocess_malay_review(text):
    if not isinstance(text, str):
        return text

    # b) Remove URLs and hyperlinks
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # g) Convert to lowercase EARLY so dictionary matching doesn't miss capitalized slang
    text = text.lower()

    # h) Translate emojis to Malay sentiment text using the external JSON map
    for emj, word in EMOJI_TO_MALAY.items():
        text = text.replace(emj, word)

    # i) Normalize repeated characters (e.g., "haukkk!" -> "hauk!")
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # Convert superscript numbers to standard digits to prevent Malaya errors
    text = text.replace('²', '2').replace('³', '3')

    # j) Normalize informal Malay & slang using REGEX to ignore attached punctuation
    for slang, formal in FORMAL_MALAY_MAP.items():
        # \b ensures we only match whole words (e.g., it replaces "je" but won't ruin "jejak")
        text = re.sub(rf'\b{slang}\b', formal, text)

    # j) Apply rule-based Malay Normalizer (Malaya)
    res = normalizer.normalize(text)
    text = res['normalize']

    # c) STRICT FILTER: Remove punctuation and non-Latin symbols
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # d) Normalize whitespace and newlines
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

print("Starting Malay text normalization pipeline...")
df_clean['cleaned_comment'] = df_clean['comment'].apply(preprocess_malay_review)

df_clean = df_clean[df_clean['cleaned_comment'].str.strip() != ''].copy()

before_dedup = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['cleaned_comment'], keep='first')
print(f"Dropped {before_dedup - len(df_clean)} rows that became duplicates post-normalization.")

vocab = Counter(' '.join(df_clean['cleaned_comment']).split())
print(f"Total vocabulary size: {len(vocab)} unique tokens")

output_path = "dataset_v3_malay_clean5.csv"
df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"Preprocessed file saved successfully as '{output_path}'. Total rows: {len(df_clean)}")