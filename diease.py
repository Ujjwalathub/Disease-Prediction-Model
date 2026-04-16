import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. LOAD DATA & COMBINE
# ==========================================
train_df = pd.read_csv(r"E:\ML\archive\Training.csv")
test_df = pd.read_csv(r"E:\ML\archive\Testing.csv")

print("Original Training data shape:", train_df.shape)
print("Original Testing data shape:", test_df.shape)

# Combine both datasets for proper stratified splitting
df_combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
print(f"Combined data shape: {df_combined.shape}\n")

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
# Handle common issue in this specific dataset format (an empty 'Unnamed: 133' column)
if 'Unnamed: 133' in df_combined.columns:
    df_combined = df_combined.drop('Unnamed: 133', axis=1)

# The target column is usually 'prognosis'
target_col = 'prognosis' if 'prognosis' in df_combined.columns else df_combined.columns[-1]
print(f"Target Column: {target_col}")

# Check unique diseases
diseases = df_combined[target_col].unique()
print(f"Number of unique diseases: {len(diseases)}")
print(f"Disease distribution:\n{df_combined[target_col].value_counts().sort_index()}\n")

# Separate features and target
X = df_combined.drop(target_col, axis=1)
y = df_combined[target_col]

# ==========================================
# 3. STRATIFIED TRAIN-TEST SPLIT
# ==========================================
# Use stratify=y to ensure each disease is proportionally represented in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,           # 20% of data goes to testing
    random_state=42,          # Ensures reproducible results
    stratify=y                # CRITICAL: Ensures equal disease distribution!
)

print("=" * 60)
print("STRATIFIED SPLIT RESULTS")
print("=" * 60)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"\nDisease representation in Training set:")
print(y_train.value_counts().sort_index())
print(f"\nDisease representation in Testing set:")
print(y_test.value_counts().sort_index())
print("=" * 60 + "\n")

# ==========================================
# 3. MODEL TRAINING
# ==========================================
print("Training Random Forest Classifier...\n")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# ==========================================
# 4. EVALUATION METRICS
# ==========================================
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("--- OVERALL METRICS ---")
print(f"Model Accuracy on Testing Data: {accuracy * 100:.2f}%")
print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
print(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}\n")

print("--- MISCLASSIFIED CLASSES (if any) ---")
# Find classes where F1 score is less than 1.0
misclassified = report_df[(report_df['f1-score'] < 1.0) & 
                          (~report_df.index.isin(['accuracy', 'macro avg', 'weighted avg']))]

if not misclassified.empty:
    print(misclassified[['precision', 'recall', 'f1-score', 'support']])
else:
    print("None! All classes were predicted perfectly in this test set.")
    
print("\nGenerating visualizations...")

# ==========================================
# 5. VISUALIZATIONS
# ==========================================


# Viz 2: Feature Importance (Top 15 Symptoms)
importances = rf.feature_importances_
indices = np.argsort(importances)[-15:] # Top 15
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices], align='center', color='coral')
plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
plt.title('Top 15 Most Important Symptoms for Prediction')
plt.xlabel('Relative Importance in Random Forest Model')
plt.tight_layout()
plt.show()

# Define top features for correlation analysis
top_features = [X_train.columns[i] for i in np.argsort(importances)[-20:]]  # Top 20

# Viz 3: Confusion Matrix Plot
plt.figure(figsize=(20, 18))
cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf.classes_, yticklabels=rf.classes_, cbar=False)
plt.title('Confusion Matrix: Disease Prediction (41 Classes)', fontsize=20)
plt.xlabel('Predicted Disease', fontsize=16)
plt.ylabel('Actual Disease', fontsize=16)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
correlation_matrix = X_train[top_features].corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Top 20 Symptoms')
plt.tight_layout()
plt.show()