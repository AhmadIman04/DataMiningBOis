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
print(df)

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

# Drop original text + embedding column
df = df.drop(columns=["customer_feedback", "customer_feedback(vector)"])

# Combine embeddings back
df = pd.concat([df, embeddings_df], axis=1)

X = df.drop(columns=["churn", "customer_id"])
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

genai.configure(api_key="API_KEY")

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
