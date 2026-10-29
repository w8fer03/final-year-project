import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel

# 1. Load Cleaned Data and Map Sentiments
df = pd.read_csv('dataset_v3_malay_clean5.csv')
X = df['cleaned_comment'].astype(str).tolist()

# Map minority class (Negative) to 1 for XGBoost scale_pos_weight targeting
y = df['sentiment'].map({'Negative': 1, 'Positive': 0}).values 

# 2. Execute Stratified Split (80:20 Example)
# You will repeat this process for your 70:30 contrasting ratio later
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    stratify=y, 
    random_state=42
)

# 3. Initialize DistilBERT Tokenizer and Model
model_name = 'distilbert-base-multilingual-cased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Move model to GPU if available for faster processing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 4. Define Batch Extraction Function
def get_cls_embeddings(text_list, batch_size=32):
    all_embeddings = []
    
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        
        # Tokenize: max_length 128, apply padding and truncation, create attention masks
        encoded = tokenizer(
            batch_texts,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move inputs to the same device as the model
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Extract features without computing gradients (saves memory)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
        # Isolate the [CLS] token (index 0) from the last hidden state
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
        
        print(f"Processed batch {i // batch_size + 1}...")
        
    return np.vstack(all_embeddings)

# 5. Extract Embeddings
print("Extracting Training Embeddings...")
X_train_embeddings = get_cls_embeddings(X_train)

print("Extracting Testing Embeddings...")
X_test_embeddings = get_cls_embeddings(X_test)

# 6. Save Embeddings to Disk
# Crucial: Save these so you don't have to rerun DistilBERT during XGBoost hyperparameter tuning
np.save('X_train_80_embeddings.npy', X_train_embeddings)
np.save('X_test_20_embeddings.npy', X_test_embeddings)
np.save('y_train_80.npy', y_train)
np.save('y_test_20.npy', y_test)

print(f"Final Train Feature Matrix Shape: {X_train_embeddings.shape}")
print(f"Final Test Feature Matrix Shape: {X_test_embeddings.shape}")