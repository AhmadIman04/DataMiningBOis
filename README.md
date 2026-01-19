

# AI-Enhanced Customer Churn Prediction 📊🤖

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green)
![Gemini AI](https://img.shields.io/badge/Gemini%20AI-Integration-purple)

---

## 📌 Project Overview

**WIE 3007 – Data Mining and Warehousing (Universiti Malaya)**

colab link : https://colab.research.google.com/drive/1rEWnthKxZfBzUNoJ8T7U_M-lvr4SLdFq?usp=sharing#scrollTo=cOGI7TucRG5Q

This project implements an end-to-end data mining pipeline to predict customer churn in a subscription-based digital service. Beyond traditional machine learning approaches, the project integrates **Generative AI** for dataset simulation and **Large Language Models (LLMs)** for feature engineering and automated business insight generation.

Both **structured data** (demographics, usage metrics) and **unstructured data** (customer feedback text) were combined to achieve high predictive performance.

---

## 👥 Group Members

| Name                                   | Matric ID | Role                                  |
| -------------------------------------- | --------- | ------------------------------------- |
| **Ahmad Iman Bin Azrul Hasni**         | 22002606  | Feature Engineering & XGBoost         |
| Luqman Nurhakim Bin Md Rapit           | 23063748  | Neural Network                        |
| Muhammad Izzhan Hakimi Bin Mohd Izzmir | 22001100  | Data Generation & Logistic Regression |
| Afzal Bin Zainol Ariffin               | 22001960  | Decision Tree                         |
| Irfan Najmi Bin Khairunizam            | 22002077  | Random Forest                         |

---

## 🚀 Key Features

### 1. AI-Simulated Dataset

* **Generator:** GPT-4o-mini
* **Dataset Size:** 2,000 realistic customer records
* **Design Logic:** Encodes causal relationships

  * e.g. High support ticket volume → higher churn probability

---

### 2. Hybrid Feature Engineering

* **Structured Features**

  * One-Hot Encoding
  * Standard Scaling
* **Unstructured Features**

  * Sentence-BERT (`all-MiniLM-L6-v2`)
  * Converted customer feedback into 384-dimensional embeddings (`embed_0` → `embed_383`)

---

### 3. Automated Insight Generation

* Integrated **Google Gemini 2.5 Flash** directly into the Python pipeline
* Automatically interprets:

  * Model performance
  * Feature importance
* Generates a **“Senior Data Scientist-style” business report**

---

## 🏆 Model Performance

Five machine learning models were trained and evaluated.
**Random Forest** achieved the highest accuracy, while **XGBoost** produced the strongest discrimination (ROC-AUC).

| Model               | Accuracy   | F1-Score   | ROC-AUC    | RMSE       |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| **Random Forest**   | **94.55%** | **0.9451** | 0.9887     | 0.2402     |
| **XGBoost**         | 94.18%     | 0.9403     | **0.9889** | **0.2165** |
| Neural Network      | 92.73%     | 0.9254     | 0.9756     | 0.2427     |
| Logistic Regression | 89.82%     | 0.8915     | 0.9716     | 0.2938     |
| Decision Tree       | 89.82%     | 0.8986     | 0.8774     | 0.3120     |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AhmadIman04/DataMiningBOis.git
cd DataMiningBOis
```

---

### 2. Install Dependencies

Ensure Python 3.10+ is installed.

```bash
pip install pandas numpy scikit-learn xgboost tensorflow sentence-transformers google-generativeai python-dotenv
```

---

### 3. Setup Environment Variables

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY="your_api_key_here"
```

---

### 4. Run the Pipeline

Ensure `clean_customer_churn_dataset.csv` is present, then run:

```bash
python main_churn_model.py
```

> Replace `main_churn_model.py` with the actual script name if different.

---

## 🧠 Business Insights

AI-assisted analysis identified the following primary churn drivers:

* **Support Ticket Volume**
  The strongest predictor across all models. High ticket frequency signals unresolved dissatisfaction.

* **Tenure & Engagement**
  New customers with low session duration are at the highest risk.

* **Customer Sentiment (Text Embeddings)**
  Negative sentiment in feedback often predicts churn before usage metrics decline.

---

## 📜 License & Disclosure

**AI Disclosure**

* Dataset generation: **GPT-4o-mini**
* Reporting & insights: **Gemini 2.5 Flash**

**Course**

* WIE 3007 – Data Mining and Warehousing
* Semester 2, Session 2025/2026

---

