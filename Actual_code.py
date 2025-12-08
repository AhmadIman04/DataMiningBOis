import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

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

###########################################################################################
#start tulis code dekat sini
#buat model

df['subscription_type'] = df['subscription_type'].map({
    'Basic': 1,
    'Standard':2,
    'Premium':3
})

df.drop(columns = ["customer_feedback","customer_id"])

print(df)