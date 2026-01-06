import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Initialize basic settings
load_dotenv()

print("✅ Environment and Libraries ready.")

df = pd.read_csv("clean_customer_churn_dataset.csv")

# Load the model once(fast)
model_slm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str) -> np.ndarray:
    """
    Returns a vector embedding for the input string using all-MiniLM-L6-v2.
    """
    return model_slm.encode(text, convert_to_numpy=True)

print("✅ Embedding function and SLM model initialized.")


df["customer_feedback(vector)"] = df["customer_feedback"].apply(get_embedding)
print(df)

###########################################################################################
#RANDOM FOREST MODEL
#buat model
# --- 5. CATEGORICAL MAPPING ---
# Mapping subscription to ordinal values and gender to binary
df['subscription_type'] = df['subscription_type'].map({'Basic': 1, 'Standard': 2, 'Premium': 3})
df['gender'] = df['gender'].map({'M': 1, 'F': 0})

# Drop raw text and ID (we have vectors and ID isn't a feature)
df.drop(columns=["customer_feedback", "customer_id"], inplace=True)
print(" Categorical data mapped and IDs dropped.")

# --- 6. FEATURE FUSION & SPLIT ---
# Convert the list of vectors into a separate DataFrame
embedding_df = pd.DataFrame(df['customer_feedback(vector)'].tolist(), index=df.index)
embedding_df.columns = [f'embed_{i}' for i in range(embedding_df.shape[1])]

# Concatenate tabular features with the new embedding features
X = pd.concat([df.drop(columns=['churn', 'customer_feedback(vector)']), embedding_df], axis=1)
y = df['churn']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(" Features fused and data split for training.")
