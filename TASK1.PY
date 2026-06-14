import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc

# ==========================================
# STEP 1: LOCATE AND LOAD DATA
# ==========================================
print("--- Step 1: Loading Dataframes ---")
path = kagglehub.dataset_download("kartik2112/fraud-detection")
train_path = os.path.join(path, "fraudTrain.csv")
test_path = os.path.join(path, "fraudTest.csv")

# We are reading a subset (nrows) first so it completes instantly on your machine.
# You can remove 'nrows' completely later to run on the full 1.2+ million records!
train_df = pd.read_csv(train_path, nrows=100000)
test_df = pd.read_csv(test_path, nrows=50000)

# ==========================================
# STEP 2: FEATURE ENGINEERING & CLEANING
# ==========================================
print("\n--- Step 2: Engineering Features ---")

def prepare_data(df):
    # Calculate distance from home location to merchant location
    df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
    
    # Extract transaction hour from timestamp string
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    
    # Frequency encoding for transaction categories
    cat_freq = df['category'].value_counts() / len(df)
    df['category_encoded'] = df['category'].map(cat_freq)
    
    # Selecting engineered features
    features = ['amt', 'distance', 'hour', 'category_encoded']
    return df[features], df['is_fraud']

X_train, y_train = prepare_data(train_df)
X_test, y_test = prepare_data(test_df)

# Scale features so numeric columns match mathematically
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# STEP 3: TRAINING EXPERIMENTS
# ==========================================
print("\n--- Step 3: Training Models (Please wait...) ---")

models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=30, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    
    results[name] = {
        "report": classification_report(y_test, y_pred, output_dict=True),
        "pr_auc": pr_auc
    }

# ==========================================
# STEP 4: PRINT FINAL REVENUE COMPARISON
# ==========================================
print("\n=========================================================================")
print(f"{'Algorithm':<25} | {'Fraud Recall':<15} | {'Fraud Precision':<18} | {'PR-AUC':<10}")
print("=========================================================================")

for name, metrics in results.items():
    fraud_key = "1" if "1" in metrics["report"] else "1.0"
    rec = metrics["report"][fraud_key]["recall"]
    prec = metrics["report"][fraud_key]["precision"]
    print(f"{name:<25} | {rec:<15.4f} | {prec:<18.4f} | {metrics['pr_auc']:<10.4f}")
print("=========================================================================")
import matplotlib.pyplot as plt

# चार्ट के लिए डेटा तैयार करना
algorithms = ['Logistic Regression', 'Decision Tree', 'Random Forest']
recall_scores = [0.7437, 0.8342, 0.8191]
precision_scores = [0.0604, 0.0849, 0.1088]

x = np.arange(len(algorithms))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, recall_scores, width, label='Fraud Recall', color='#1f77b4')
rects2 = ax.bar(x + width/2, precision_scores, width, label='Fraud Precision', color='#ff7f0e')

ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison (Fraud Detection)')
ax.set_xticks(x)
ax.set_xticklabels(algorithms)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
# चार्ट को इमेज के रूप में सेव करना
plt.savefig('model_comparison_chart.png')
print("\n[SUCCESS] Explanation chart saved as 'model_comparison_chart.png'!")
plt.show()