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

# Load model once (fast)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str) -> np.ndarray:
    """
    Returns a vector embedding for the input string
    using all-MiniLM-L6-v2.
    """
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding

df["customer_feedback(vector)"] = df["customer_feedback"].apply(get_embedding)
print(df)

###########################################################################################
#start tulis code dekat sini
#buat model

