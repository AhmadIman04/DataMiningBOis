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

###########################################################################################
#start tulis code dekat sini
#buat model

#--------------------------------Random Forest ----------------------------------

df['subscription_type'] = df['subscription_type'].map({
    'Basic': 1,
    'Standard': 2,
    'Premium': 3
})

df.drop(columns = ["customer_feedback", "customer_id"], inplace=True)

df['gender'] = df['gender'].map({
    'M': 1,
    'F': 0,
})

load_dotenv() 

def gemini_reply(prompt):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    # Updated to stable gemini-1.5-flash to fix 404 error
    model_ai = genai.GenerativeModel("gemini-1.5-flash") 
    response = model_ai.generate_content(prompt)
    return response.text

embedding_df = pd.DataFrame(df['customer_feedback(vector)'].tolist(), index=df.index)
embedding_df.columns = [f'embed_{i}' for i in range(embedding_df.shape[1])]

# Concatenate tabular features with the new embedding features
X = pd.concat([df.drop(columns=['churn', 'customer_feedback(vector)']), embedding_df], axis=1)
y = df['churn']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest initialization
model_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

model_rf.fit(X_train, y_train)

# Predictions
y_pred = model_rf.predict(X_test)
y_prob = model_rf.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
rmse = np.sqrt(mean_squared_error(y_test, y_prob)) # RMSE on probabilities

# Extract Feature Importance (RF Gini Importance)
importances = model_rf.feature_importances_
feature_importance_df = pd.DataFrame(
    data=importances, 
    index=X.columns, 
    columns=["score"]
).sort_values(by="score", ascending=False)

top_features = feature_importance_df.head(10).to_string()

print(f"Metrics Calculated:\nAccuracy: {accuracy}\nF1: {f1}\nROC-AUC: {roc_auc}\nRMSE: {rmse}")

# Define the Prompt
analysis_prompt = f"""
You are a Senior Data Scientist. I have trained a Random Forest model to predict customer churn. 
Here are the model performance metrics and the top contributing features.

**Model Metrics:**
- Accuracy: {accuracy:.4f}
- F1-Score: {f1:.4f}
- ROC-AUC Score: {roc_auc:.4f}
- RMSE (on probabilities): {rmse:.4f}

**Top Features Driving Decisions:**
{top_features}

**Context:**
- Features starting with 'embed_' come from the vector embeddings of customer feedback text.
- Other features are standard demographic/usage data (age, monthly_spend, etc).

**Task:**
1. Summarize the model's performance. Is it good?
2. Interpret the feature importance. Does the text feedback (embedding features) play a bigger role than the structured data (like age/spend)?
3. Provide actionable business insights based on these findings.
"""

try:
    insights = gemini_reply(analysis_prompt)
    print("\n" + "="*50)
    print("GEMINI INSIGHTS REPORT")
    print("="*50)
    print(insights)
except Exception as e:
    print(f"Error calling Gemini: {e}")

#----------------------------------------------------------------------------------------