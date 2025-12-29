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

#----------------------------------------------------------------------------------------