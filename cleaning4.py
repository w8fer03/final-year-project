import pandas as pd
import re
import malaya
from collections import Counter

# ==============================================================================
# h) EMOJI TO MALAY SENTIMENT MAPPING
# Maps emojis directly to Malay sentiment keywords (Lim et al., 2024)
# ==============================================================================
EMOJI_TO_MALAY = {
    # Positive expressions (suka, gembira, bagus)
    "❤️": " suka ", "💖": " suka ", "💕": " suka ", "🥰": " suka ", "😍": " suka ",
    "👍": " bagus ", "👏": " bagus ", "🔥": " mantap ", "✨": " cantik ",
    "😊": " gembira ", "😀": " gembira ", "😁": " gembira ", "😄": " gembira ",
    
    # Negative expressions (marah, kecewa, sedih, buruk)
    "😡": " marah ", "😠": " marah ", "🤬": " marah ",
    "😭": " sedih ", "😢": " sedih ", "😞": " kecewa ", "😔": " kecewa ",
    "👎": " teruk ", "🤮": " teruk ", "🤢": " teruk ", "💔": " kecewa ",
    "❌": " salah ", "🚫": " tidak ",
    
    # Mild / Contextual (default sentiment assignment)
    "😅": " malu ", "🥺": " merayu "
}

# ==============================================================================
# j) STANDARDIZE INFORMAL MALAY (BAHASA ROJAK TO BAKU)
# Maps slang, colloquialisms, and informal variations to formal Malay equivalents
# ==============================================================================
FORMAL_MALAY_MAP = {
    # Positive sentiment standardization
    "mantap": "bagus",
    "padu": "bagus",
    "power": "bagus",
    "sedap": "bagus",
    "terbaik": "bagus",
    
    # Negative sentiment standardization
    "hauk": "teruk",
    "kureng": "kurang",
    "barai": "rosak",
    "rabak": "rosak",
    "koyak": "rosak",

    # Colloquialisms & internet shorthand
    "kak": "kakak",
    "akak": "kakak",
    "taknak": "tidak mahu",
    "xnak": "tidak mahu",
    "xda": "tidak ada",
    "xdak": "tidak ada",
    "takde": "tidak ada",
    "takdela": "tidak adalah",
    "kecik": "kecil",
    "jelah": "sahaja lah",
    "je": "sahaja",
    "lah": "lah",
    "la": "lah",
    "kat": "dekat",
    "kt": "dekat",
    "yg": "yang",
    "dgn": "dengan",
    "mmg": "memang",
    "sbb": "sebab",
    "nk": "nak",
    "tak": "tidak",
    "x": "tidak"
}

# 1. Load Malaya's rule-based normalizer (Ariffin & Tiun, 2020)
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

    # h) Translate emojis to Malay sentiment text
    for emj, word in EMOJI_TO_MALAY.items():
        text = text.replace(emj, word)

    # i) Normalize repeated characters (e.g., "baggussss" -> "bagus", "besttt" -> "best")
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # j) Normalize informal Malay & slang using the custom Malay dictionary
    tokens = text.split()
    tokens = [FORMAL_MALAY_MAP.get(tok.lower(), tok) for tok in tokens]
    text = " ".join(tokens)

    #Convert superscript numbers to standard digits so Malaya doesn't crash
    text = text.replace('²', '2').replace('³', '3')

    # j) Apply rule-based Malay Normalizer (Malaya) for morphology & spelling
    res = normalizer.normalize(text)
    text = res['normalize']

    # g) Convert to lowercase
    text = text.lower()

    # c) Remove punctuation, non-Latin symbols, and unmapped emojis
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # d) Normalize whitespace and newlines
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

print("Starting Malay text normalization pipeline...")
df_clean['cleaned_comment'] = df_clean['comment'].apply(preprocess_malay_review)

# Remove any reviews that became empty strings post-cleaning
df_clean = df_clean[df_clean['cleaned_comment'].str.strip() != ''].copy()

# Deduplicate identical normalized comments
before_dedup = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['cleaned_comment'], keep='first')
print(f"Dropped {before_dedup - len(df_clean)} rows that became duplicates post-normalization.")

# Vocab analysis
vocab = Counter(' '.join(df_clean['cleaned_comment']).split())
print(f"Total vocabulary size: {len(vocab)} unique tokens")

# Save cleaned output
output_path = "dataset_v3_malay_standardized.csv"
df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"Preprocessed file saved successfully as '{output_path}'. Total rows: {len(df_clean)}")