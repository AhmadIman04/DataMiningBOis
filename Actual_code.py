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

# --- 7. RANDOM FOREST MODELING ---
print("Training Random Forest... this might take a moment.")
model_rf = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    class_weight='balanced'
)

model_rf.fit(X_train, y_train)
print("Random Forest model training complete.")

# --- 8. EVALUATION & AI INSIGHTS ---
# Predictions
y_pred = model_rf.predict(X_test)
y_prob = model_rf.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Feature Importance
importances = model_rf.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X.columns, 'score': importances}).sort_values(by="score", ascending=False)
top_features = feature_importance_df.head(10).to_string(index=False)

print(f"\n📊 METRICS:\nAccuracy: {accuracy:.4f}\nF1: {f1:.4f}\nROC-AUC: {roc_auc:.4f}")

def gemini_reply(prompt):
    # Configure using the environment variable
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Use 'gemini-1.5-flash-latest' to avoid the 404 error
    model_ai = genai.GenerativeModel("gemini-1.5-flash-latest")
    
    response = model_ai.generate_content(prompt)
    return response.text

analysis_prompt = f"""
Senior Data Scientist Report:
Random Forest Churn Model Results:
- Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}
- Top Drivers of Churn:
{top_features}

Explain if the sentiment (embeddings) mattered more than spending, and give 3 business actions.
"""

try:
    print("\n🤖 GEMINI ANALYSIS:")
    print("="*30)
    print(gemini_reply(analysis_prompt))
    print("="*30)
except Exception as e:
    # If it still fails, it might be an API versioning issue
    print(f"AI Error: {e}")
    print("Tip: Check if your API Key is valid and billing/quota is active.")