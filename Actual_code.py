import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyBUgU-tOXikctC09kdmU9UWSd60VnyyU-M" 

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
#DECISION TREE MODEL FOR CUSTOMER CHURN PREDICTION
print("Start of the decision tree model code")
# 1. PREPARE THE FEATURES (X) AND TARGET (y)

# A. Handle the Embeddings
# We need to expand the list/array in "customer_feedback(vector)" into separate columns
embedding_df = pd.DataFrame(df["customer_feedback(vector)"].tolist(), index=df.index)
# Rename these new columns to something like embed_0, embed_1, etc.
embedding_df.columns = [f"embed_{i}" for i in range(embedding_df.shape[1])]

# B. Handle Categorical Data (One-Hot Encoding)
# Select numerical and categorical features
features = df[['age', 'gender', 'subscription_type', 'tenure_months', 
               'monthly_spend', 'support_tickets', 'avg_session_time']]

# Convert 'gender' and 'subscription_type' to numbers (0s and 1s)
features_encoded = pd.get_dummies(features, columns=['gender', 'subscription_type'], drop_first=True)

# C. Combine everything into X
# Join the numerical/categorical features with the embedding features
X = pd.concat([features_encoded, embedding_df], axis=1)

# D. Define Target
y = df['churn']

# 2. SPLIT DATA (Training and Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. BUILD THE DECISION TREE MODEL
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5) # max_depth limits tree size to prevent overfitting
dt_model.fit(X_train, y_train)

# 4. PREDICT AND EVALUATE
y_pred = dt_model.predict(X_test)

y_prob = dt_model.predict_proba(X_test)[:, 1] # Probability for ROC-AUC

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("-" * 30)


##FEATURE IMPORTANCE EXTRACTION

feature_names = X.columns
importances = dt_model.feature_importances_
# Get indices of top 10 features
indices = np.argsort(importances)[::-1][:10]

top_features_list = []
for i in indices:
    feat_name = feature_names[i]
    score = importances[i]
    top_features_list.append(f"{feat_name}: {score:.4f}")

top_features_str = "\n".join(top_features_list)
print("Top 10 Features Driving Churn:\n" + top_features_str)


##GEMINI LLM INTEGRATION
def get_gemini_analysis(api_key, accuracy, roc, top_feats):
    if api_key == "YOUR_GEMINI_API_KEY_HERE":
        print("\n[!] Please insert your Gemini API Key to get the LLM analysis.")
        return

    try:
        genai.configure(api_key=api_key)
        # using gemini-1.5-flash because it is fast and cheap/free tier eligible
        llm_model = genai.GenerativeModel('gemini-2.5-flash') 

        prompt = f"""
        You are an expert Data Scientist. I have trained a Customer Churn Prediction model (Decision Tree).
        
        Here are the evaluation metrics:
        - Accuracy: {accuracy:.2f}
        - ROC-AUC Score: {roc:.2f}
        
        Here are the top 10 most important features driving the model:
        {top_feats}
        
        Note: Features starting with 'embed_' refer to dimensions in the customer feedback text vector (semantic meaning of their review).
        Features like 'support_tickets' or 'monthly_spend' are numerical.
        
        Please provide a brief report:
        1. Summarize the model's performance (is it good based on ROC-AUC?).
        2. Interpret the important features. What is driving churn the most?
        3. Provide 2 actionable business insights based on these findings.
        """
        
        print("\n" + "="*40)
        print("GENERATING INSIGHTS WITH GEMINI...")
        print("="*40)
        
        response = llm_model.generate_content(prompt)
        print(response.text)
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")

# Run the Gemini Analysis
get_gemini_analysis(GOOGLE_API_KEY, acc, roc_auc, top_features_str)