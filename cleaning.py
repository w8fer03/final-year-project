import pandas as pd
import re
import malaya

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
    
    # NEW: Force the text to lowercase before the strict filter
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

# Save your fully cleaned dataset to a new CSV file
df_clean.to_csv("dataset_v3_malaya_cleaned.csv", index=False, encoding='utf-8-sig')
print(f"Data cleaning complete. Final dataset contains {len(df_clean)} rows.")