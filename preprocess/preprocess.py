# -*- coding: utf-8 -*-
import os
import json
import re
import urllib.request
import difflib
import pandas as pd
from underthesea import word_tokenize

class ShopeeReviewPreprocessor:
    """
    Advanced preprocessing pipeline for Vietnamese Shopee reviews.
    Implements:
    1. Basic text cleaning (lowercasing, punctuation removal).
    2. Custom teencode & abbreviation normalization (teencode.json).
    3. Edit distance-based typo correction using a standard Vietnamese dictionary.
    4. Word segmentation using Underthesea.
    """
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir
        
        self.teencode_path = os.path.join(data_dir, "teencode.json")
        self.dict_path = os.path.join(data_dir, "vietnamese_dictionary.txt")
        
        # Load custom abbreviations & slang dictionary
        self.teencode_dict = self._load_teencode()
        
        # Load standard Vietnamese vocabulary
        self.vocab = self._load_vocabulary()
        
    def _load_teencode(self):
        if os.path.exists(self.teencode_path):
            try:
                with open(self.teencode_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(f"[✓] Custom teencode dictionary loaded: {len(data)} rules.")
                    return data
            except Exception as e:
                print(f"[!] Error loading teencode.json: {e}")
        return {}
        
    def _load_vocabulary(self):
        # Download standard Vietnamese wordlist if not exists
        if not os.path.exists(self.dict_path):
            url = "https://vietnamese-wordlist.duyet.net/Viet74K.txt"
            print(f"[*] Downloading standard Vietnamese dictionary from: {url}...")
            try:
                # Add headers to avoid potential HTTP block
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response, open(self.dict_path, 'wb') as out_file:
                    out_file.write(response.read())
                print("[✓] Dictionary downloaded successfully!")
            except Exception as e:
                print(f"[!] Failed to download wordlist ({e}). Using robust fallback vocabulary.")
                # Core fallback vocabulary of common Vietnamese words to ensure it runs offline
                fallback_words = [
                    "không", "sản", "phẩm", "được", "tốt", "đẹp", "giao", "hàng", "cửa", "hàng", 
                    "nhân", "viên", "mua", "dùng", "ổn", "nhẹ", "sáng", "mắt", "mỏi", "nhanh", 
                    "chậm", "tiện", "lợi", "chất", "lượng", "đóng", "gói", "cẩn", "thận", 
                    "thích", "ghét", "hài", "lòng", "tuyệt", "vời", "chuẩn", "mô", "tả",
                    "cảm", "ứng", "điều", "khiển", "từ", "xa", "đèn", "led", "màn", "hình"
                ]
                with open(self.dict_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(fallback_words))
                    
        vocab = set()
        if os.path.exists(self.dict_path):
            try:
                with open(self.dict_path, "r", encoding="utf-8") as f:
                    for line in f:
                        word = line.strip().lower()
                        if word:
                            vocab.add(word)
                print(f"[✓] Standard vocabulary loaded: {len(vocab)} words.")
            except Exception as e:
                print(f"[!] Error reading dictionary: {e}")
        return vocab

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Remove punctuation, keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize_teencode(self, text):
        words = text.split()
        normalized = [self.teencode_dict.get(w, w) for w in words]
        return " ".join(normalized)

    def correct_spelling(self, text, cutoff=0.85):
        """
        Correct spelling errors (typos) using Edit Distance.
        Checks if each word is in standard Vietnamese vocabulary. 
        If not, replaces it with the closest match having high similarity.
        """
        words = text.split()
        corrected_words = []
        
        for w in words:
            # Keep numbers, very short terms (<= 2 chars like 'đi', 'an'), and valid words
            if w.isdigit() or w in self.vocab or len(w) <= 2:
                corrected_words.append(w)
                continue
                
            # Use Python standard library difflib for zero-dependency optimized match
            matches = difflib.get_close_matches(w, self.vocab, n=1, cutoff=cutoff)
            if matches:
                corrected_words.append(matches[0])
            else:
                corrected_words.append(w)
                
        return " ".join(corrected_words)

    def preprocess_comment(self, comment):
        if not isinstance(comment, str) or not comment.strip():
            return "", "", ""
            
        # 1. Clean comment
        clean = self.clean_text(comment)
        if not clean:
            return "", "", ""
            
        # 2. Normalize teencode & acronyms (Pre-Tokenization)
        normalized = self.normalize_teencode(clean)
        
        # 3. Spellcheck & correct typos using Edit Distance (Pre-Tokenization)
        corrected = self.correct_spelling(normalized)
        
        # 4. Tokenization (Word Segmentation)
        tokens = word_tokenize(corrected, format="text")
        
        return clean, corrected, tokens

    def process_dataframe(self, df):
        processed_df = df.copy()
        
        # Split datetime into date and time
        if 'datetime' in processed_df.columns:
            processed_df[['date', 'time']] = processed_df['datetime'].str.split(' ', expand=True)
            
        # Map variant to category
        if 'variant' in processed_df.columns:
            processed_df['category'] = processed_df['variant'].fillna("")
            
        # Remove rows without comments
        processed_df = processed_df.dropna(subset=['comment'])
        
        # Remove duplicate reviews
        if 'cmtid' in processed_df.columns:
            processed_df = processed_df.drop_duplicates(subset=['cmtid'])
        else:
            processed_df = processed_df.drop_duplicates(subset=['user', 'comment'])
            
        print(f"[*] Preprocessing {len(processed_df)} reviews...")
        
        cleans, correcteds, tokens_list = [], [], []
        for i, cmt in enumerate(processed_df['comment']):
            clean, corrected, tokens = self.preprocess_comment(cmt)
            cleans.append(clean)
            correcteds.append(corrected)
            tokens_list.append(tokens)
            
        processed_df['comment_clean'] = cleans
        processed_df['comment_corrected'] = correcteds
        processed_df['tokens_normalized'] = tokens_list
        
        print("[✓] Preprocessing pipeline finished successfully!")
        return processed_df
