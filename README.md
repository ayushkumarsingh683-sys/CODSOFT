# CODSOFT - Data Science Internship

This repository contains the machine learning projects completed during my Data Science Virtual Internship at **CodSoft**.

---

## 💳 Task 1: Credit Card Fraud Detection

### 📌 Project Overview
Developed a binary classification pipeline to detect fraudulent credit card transactions from a highly imbalanced dataset of **1.2 Million+ records** (Sparkov simulator).

### 🛠️ Feature Engineering
* **Geographical Distance:** Calculated distance between customer home and merchant locations.
* **Temporal Context:** Extracted transaction `hour` to flag late-night anomalies.
* **Frequency Encoding:** Converted high-cardinality `category` text into numeric probability signals.

### 📊 Model Performance Matrix
Optimized using balanced class weights and evaluated on **Fraud Recall** and **PR-AUC**:

| Algorithm | Fraud Recall | Fraud Precision | PR-AUC |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 0.7437 | 0.0604 | 0.1211 |
| **Decision Tree** | 0.8342 | 0.0849 | 0.3650 |
| **Random Forest (Winner)** | **0.8191** | **0.1088** | **0.4041** |

### 💡 Key Conclusion
**Random Forest** is the best production model. While the Decision Tree had a slightly higher Recall, Random Forest minimized false alarms, delivering the highest operational **Precision (10.88%)** and best-in-class **PR-AUC (0.4041)**.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib
