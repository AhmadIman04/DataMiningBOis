import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import google.generativeai as genai
import os

# Load .env.local

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


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

#--------------------------------Neural Network ----------------------------------

# Create a copy of df for neural network training
df_nn = df.copy()

embedding_df = pd.DataFrame(df_nn['customer_feedback(vector)'].tolist(), index=df_nn.index)
embedding_df.columns = [f'embed_{i}' for i in range(embedding_df.shape[1])]

df_nn.drop(columns=['customer_feedback', 'customer_id', 'customer_feedback(vector)'], inplace=True)

df_nn = pd.concat([df_nn, embedding_df], axis=1)

# Feature Engineering (df_nn) 
df_nn = pd.get_dummies(df_nn, columns=['gender', 'subscription_type'], drop_first=True)

scaler = StandardScaler()

num_cols = df_nn.select_dtypes(include=[np.number]).columns.tolist()
if 'churn' in num_cols:
    num_cols.remove('churn')

df_nn[num_cols] = scaler.fit_transform(df_nn[num_cols])

X_nn = df_nn.drop(columns=['churn'])
y_nn = df_nn['churn']

X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_nn, y_nn, test_size=0.2, random_state=42)

#Neural Network Builder
def build_neural_network(input_shape):
    """Builds and compiles a simple feed-forward neural network.

    Args:
        input_shape (tuple): Shape of the input features (e.g., (n_features,)).

    Returns:
        tf.keras.Model: A compiled Keras Sequential model.
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

#Train Neural Network
input_dim = X_train_nn.shape[1]
model_nn = build_neural_network((input_dim,))

# Fit the model
history_nn = model_nn.fit(
    X_train_nn,
    y_train_nn,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_nn, y_test_nn)
)


# Evaluation

y_prob_nn = model_nn.predict(X_test_nn).ravel()
y_pred_nn = (y_prob_nn > 0.5).astype(int)

# Metrics
acc_nn = accuracy_score(y_test_nn, y_pred_nn)
f1_nn = f1_score(y_test_nn, y_pred_nn)
roc_auc_nn = roc_auc_score(y_test_nn, y_prob_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test_nn, y_prob_nn))

print("\nNeural Network Metrics:")
print(f"Accuracy: {acc_nn:.4f}")
print(f"F1-Score: {f1_nn:.4f}")
print(f"ROC-AUC: {roc_auc_nn:.4f}")
print(f"RMSE (probabilities): {rmse_nn:.4f}")


# Prepare AI analysis prompt for Neural Network
accuracy_nn = acc_nn
analysis_prompt_nn = f"""
You are a Senior Data Scientist.

Model: Feed-forward Neural Network for customer churn prediction.

Metrics:
- Accuracy: {accuracy_nn:.4f}
- F1-Score: {f1_nn:.4f}
- ROC-AUC: {roc_auc_nn:.4f}
- RMSE (probabilities): {rmse_nn:.4f}

Questions:
1) Evaluate: Is this Neural Network performing well for a churn prediction task?
2) Compare: How would this likely compare to simple linear models (e.g., logistic regression)?
3) Insights: Given this model combines numerical features and `embed_` text-derived features, what business advantages does this more complex modeling approach provide?
"""


#--------------------------------XGBoost ----------------------------------

df['subscription_type'] = df['subscription_type'].map({
    'Basic': 1,
    'Standard':2,
    'Premium':3
})

df.drop(columns = ["customer_feedback","customer_id"], inplace=True)

df['gender'] = df['gender'].map({
    'M': 1,
    'F':0,
})


load_dotenv() 

def gemini_reply(prompt):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

embedding_df = pd.DataFrame(df['customer_feedback(vector)'].tolist(), index=df.index)
embedding_df.columns = [f'embed_{i}' for i in range(embedding_df.shape[1])]

# Concatenate tabular features with the new embedding features
X = pd.concat([df.drop(columns=['churn', 'customer_feedback(vector)']), embedding_df], axis=1)
y = df['churn']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model_xgb = xgb.XGBClassifier(
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

model_xgb.fit(X_train, y_train)

# Predictions
y_pred = model_xgb.predict(X_test)
y_prob = model_xgb.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
rmse = np.sqrt(mean_squared_error(y_test, y_prob)) # RMSE on probabilities

# Extract Feature Importance
feature_important = model_xgb.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

feature_importance_df = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
top_features = feature_importance_df.head(10).to_string()

print(f"Metrics Calculated:\nAccuracy: {accuracy}\nF1: {f1}\nROC-AUC: {roc_auc}\nRMSE: {rmse}")


# Define the Prompt
analysis_prompt = f"""
You are a Senior Data Scientist. I have trained an XGBoost model to predict customer churn. 
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


try:
    insights_nn = gemini_reply(analysis_prompt_nn)
    print("\n" + "="*50)
    print("=== NEURAL NETWORK AI INSIGHTS ===")
    print("="*50)
    print(insights_nn)
except Exception as e:
    print(f"Error calling Gemini for NN: {e}")

#-------------------------Logistic Regression--------------------------------

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Expand embedding column
embedding_dim = len(df["customer_feedback(vector)"].iloc[0])

embeddings_df = pd.DataFrame(
    df["customer_feedback(vector)"].tolist(),
    index=df.index,
    columns=[f"emb_{i}" for i in range(embedding_dim)]
)

# Combine embeddings back
df = pd.concat([df, embeddings_df], axis=1)

X = df.drop(columns=["churn", "customer_id"], errors='ignore')
y = df["churn"]

numeric_features = [
    "age",
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "avg_session_time"
]

categorical_features = [
    "gender",
    "subscription_type"
]

embedding_features = [col for col in X.columns if col.startswith("emb_")]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("emb", StandardScaler(), embedding_features)
    ]
)

log_reg = LogisticRegression(
    penalty="l2",
    solver="saga",
    max_iter=5000,
    class_weight="balanced",
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", log_reg)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_prob),
    "rmse": np.sqrt(mean_squared_error(y_test, y_prob))
}

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
coefficients = pipeline.named_steps["classifier"].coef_[0]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Abs_Coefficient": np.abs(coefficients)
}).sort_values(by="Abs_Coefficient", ascending=False)

# Separate business features from embeddings
business_features = importance_df[
    ~importance_df["Feature"].str.contains("emb_")
]

print(business_features)

genai.configure(api_key=GOOGLE_API_KEY)

llm = genai.GenerativeModel("gemini-2.5-flash")

prompt = f"""
You are an expert data scientist evaluating a customer churn prediction model.

MODEL:
- Logistic Regression with L2 regularization
- Sentence-BERT embeddings used for customer feedback
- Binary classification (churn)

EVALUATION METRICS:
Accuracy: {metrics['accuracy']:.4f}
F1-score: {metrics['f1_score']:.4f}
ROC-AUC: {metrics['roc_auc']:.4f}
RMSE: {metrics['rmse']:.4f}

TOP FEATURE COEFFICIENTS (Logistic Regression):
{business_features}

TASK:
1. Summarise the model performance using the provided metrics.
2. Interpret what the results imply about churn prediction quality.
3. Explain how the top features influence churn.
4. Comment on the role of customer feedback embeddings.
5. Provide business and analytical insights.

Write in a formal, academic tone suitable for a final-year project or research report.
"""

response = llm.generate_content(prompt)

print(response.text)

#-------------------------Decision Tree Model-------------------------------------
print("\n" + "="*50)
print("Start of the decision tree model code")
print("\n" + "="*50)

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

#--------------------------------Random Forest ----------------------------------


# 1. IMPORT NECESSARY LIBRARIES FOR THIS SECTION
from sklearn.ensemble import RandomForestClassifier

# 2. RELOAD DATA 
df_rf = pd.read_csv("clean_customer_churn_dataset.csv")

# 3. VECTORIZATION
print("Generating Embeddings for Random Forest...")
df_rf["customer_feedback(vector)"] = df_rf["customer_feedback"].apply(get_embedding)

# 4. PREPROCESSING
df_rf['subscription_type'] = df_rf['subscription_type'].map({'Basic': 1, 'Standard': 2, 'Premium': 3})
df_rf['gender'] = df_rf['gender'].map({'M': 1, 'F': 0})

# Drop columns not needed for training
df_rf.drop(columns=["customer_feedback", "customer_id"], inplace=True)

# Expand embeddings into separate columns
embedding_df_rf = pd.DataFrame(df_rf['customer_feedback(vector)'].tolist(), index=df_rf.index)
embedding_df_rf.columns = [f'embed_{i}' for i in range(embedding_df_rf.shape[1])]

# Concatenate tabular features with the new embedding features
X_rf = pd.concat([df_rf.drop(columns=['churn', 'customer_feedback(vector)']), embedding_df_rf], axis=1)
y_rf = df_rf['churn']

# 5. SPLIT DATA
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# 6. INITIALIZE AND TRAIN RANDOM FOREST
print("Training Random Forest...")
model_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
model_rf.fit(X_train_rf, y_train_rf)

# 7. PREDICTIONS & METRICS
y_pred_rf = model_rf.predict(X_test_rf)
y_prob_rf = model_rf.predict_proba(X_test_rf)[:, 1]

accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
f1_rf = f1_score(y_test_rf, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test_rf, y_prob_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_rf, y_prob_rf))

print(f"Random Forest Metrics:\nAccuracy: {accuracy_rf:.4f}\nF1: {f1_rf:.4f}\nROC-AUC: {roc_auc_rf:.4f}\nRMSE: {rmse_rf:.4f}")

# 8. FEATURE IMPORTANCE
importances_rf = model_rf.feature_importances_
feature_importance_rf = pd.DataFrame(
    data=importances_rf, 
    index=X_rf.columns, 
    columns=["score"]
).sort_values(by="score", ascending=False)

top_features_rf = feature_importance_rf.head(10).to_string()

# 9. GEMINI AI ANALYSIS (Random Forest Specific)
def get_rf_gemini_analysis(api_key, accuracy, f1, roc, top_feats):
    if not api_key:
        print("\n[!] Gemini API Key missing.")
        return

    try:
        genai.configure(api_key=api_key)
        # Using the same model version as the rest of the file
        llm_model = genai.GenerativeModel('gemini-2.5-flash') 

        prompt = f"""
        You are a Senior Data Scientist. I have trained a Random Forest model to predict customer churn. 
        Here are the model performance metrics and the top contributing features.

        **Model Metrics:**
        - Accuracy: {accuracy:.4f}
        - F1-Score: {f1:.4f}
        - ROC-AUC Score: {roc:.4f}

        **Top Features Driving Decisions:**
        {top_feats}

        **Task:**
        1. Summarize the Random Forest model's performance.
        2. Compare the feature importance: Do the embedding features (text sentiment) matter more than structured data?
        3. Provide actionable business insights.
        """
        
        print("\n" + "="*50)
        print("GEMINI INSIGHTS REPORT (RANDOM FOREST)")
        print("="*50)
        
        response = llm_model.generate_content(prompt)
        print(response.text)
        
    except Exception as e:
        print(f"Error calling Gemini for RF: {e}")

# Run the analysis using the global API key defined at the top of the file
get_rf_gemini_analysis(GOOGLE_API_KEY, accuracy_rf, f1_rf, roc_auc_rf, top_features_rf)