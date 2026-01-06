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
print("✅ Step 5: Categorical data mapped and IDs dropped.")
