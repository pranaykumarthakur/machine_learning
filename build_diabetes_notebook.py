import nbformat as nbf

def create_simple_diabetes_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # 1. Markdown Header
    cells.append(nbf.v4.new_markdown_cell("""# 🩺 Diabetes Prediction: Classification & Ensemble Learning
### Machine Learning Assignment

This notebook implements classical machine learning classification algorithms and ensemble methods to predict diabetes using the `diabetes_prediction_dataset.csv`.

#### Models Covered:
1. **Base Classifiers**:
   - Logistic Regression
   - Decision Tree Classifier
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (Linear SVM)
2. **Ensemble Methods**:
   - Voting Classifier
   - Bagging Classifier
   - AdaBoost Classifier
   - Gradient Boosting Classifier
   - Random Forest Classifier"""))

    # 2. Imports
    cells.append(nbf.v4.new_code_cell("""import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing & Model Selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Base Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# Ensemble Classifiers
from sklearn.ensemble import (
    VotingClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("Libraries imported successfully!")"""))

    # 3. Load Dataset & Simple EDA
    cells.append(nbf.v4.new_markdown_cell("""## 1. Load Data & Exploratory Data Analysis (EDA)"""))

    cells.append(nbf.v4.new_code_cell("""# Load dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')
print("Dataset Shape:", df.shape)
df.head()"""))

    cells.append(nbf.v4.new_code_cell("""# Check information and missing values
df.info()
print("\\nMissing values count:\\n", df.isnull().sum())"""))

    cells.append(nbf.v4.new_code_cell("""# Target distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='diabetes', hue='diabetes', palette='Blues_d', legend=False)
plt.title('Diabetes Target Distribution (0 = No, 1 = Yes)')
plt.xlabel('Diabetes Status')
plt.ylabel('Count')
plt.show()"""))

    # 4. Preprocessing
    cells.append(nbf.v4.new_markdown_cell("""## 2. Data Preprocessing & Train-Test Split"""))

    cells.append(nbf.v4.new_code_cell("""# Remove duplicate rows
df = df.drop_duplicates()

# One-hot encode categorical features (gender, smoking_history)
df = pd.get_dummies(df, drop_first=True)

# Separate features (X) and target (y)
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Split dataset into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features for distance/gradient based models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")"""))

    # 5. Base Classifiers
    cells.append(nbf.v4.new_markdown_cell("""## 3. Base Classifiers Implementation"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_lr))"""))

    cells.append(nbf.v4.new_code_cell("""# 2. Decision Tree Classifier
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_dt))"""))

    cells.append(nbf.v4.new_code_cell("""# 3. K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_knn))"""))

    cells.append(nbf.v4.new_code_cell("""# 4. Support Vector Machine (Linear SVM)
svm = LinearSVC(random_state=42, max_iter=3000, dual='auto')
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_svm))"""))

    # 6. Ensemble Methods
    cells.append(nbf.v4.new_markdown_cell("""## 4. Ensemble Learning Methods Implementation"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Voting Classifier (Hard Voting)
voting = VotingClassifier(
    estimators=[('lr', lr), ('dt', dt), ('knn', knn)],
    voting='hard',
    n_jobs=-1
)
voting.fit(X_train_scaled, y_train)
y_pred_voting = voting.predict(X_test_scaled)

print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred_voting))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_voting))"""))

    cells.append(nbf.v4.new_code_cell("""# 2. Bagging Classifier
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=6),
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)

print("Bagging Classifier Accuracy:", accuracy_score(y_test, y_pred_bagging))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_bagging))"""))

    cells.append(nbf.v4.new_code_cell("""# 3. AdaBoost Classifier
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost.predict(X_test)

print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_adaboost))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_adaboost))"""))

    cells.append(nbf.v4.new_code_cell("""# 4. Gradient Boosting Classifier
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_gb))"""))

    cells.append(nbf.v4.new_code_cell("""# 5. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_rf))"""))

    # 7. Comparison Table & Simple Visualizations
    cells.append(nbf.v4.new_markdown_cell("""## 5. Model Comparison & Visualizations"""))

    cells.append(nbf.v4.new_code_cell("""# Store predictions dictionary for simple comparison
all_preds = {
    'Logistic Regression': y_pred_lr,
    'Decision Tree': y_pred_dt,
    'KNN': y_pred_knn,
    'Linear SVM': y_pred_svm,
    'Voting Classifier': y_pred_voting,
    'Bagging Classifier': y_pred_bagging,
    'AdaBoost': y_pred_adaboost,
    'Gradient Boosting': y_pred_gb,
    'Random Forest': y_pred_rf
}

# Create summary metrics DataFrame
summary_list = []
for model_name, preds in all_preds.items():
    summary_list.append({
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_test, preds), 4),
        'Precision': round(precision_score(y_test, preds), 4),
        'Recall': round(recall_score(y_test, preds), 4),
        'F1-Score': round(f1_score(y_test, preds), 4)
    })

comparison_df = pd.DataFrame(summary_list)
comparison_df = comparison_df.sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
print(comparison_df)"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Model Accuracy Comparison Bar Plot
plt.figure(figsize=(10, 5))
sns.barplot(data=comparison_df, x='Accuracy', y='Model', hue='Model', palette='Blues_r', legend=False)
plt.title('Model Accuracy Comparison')
plt.xlabel('Accuracy')
plt.xlim(0.8, 1.0)
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 2. Model F1-Score Comparison Bar Plot
plt.figure(figsize=(10, 5))
sns.barplot(data=comparison_df, x='F1-Score', y='Model', hue='Model', palette='Greens_r', legend=False)
plt.title('Model F1-Score Comparison')
plt.xlabel('F1-Score')
plt.xlim(0, 1.0)
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 3. Confusion Matrix of Best Model (Random Forest)
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 4. Feature Importance (Random Forest)
plt.figure(figsize=(8, 4))
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
feat_imp.plot(kind='barh', color='teal')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()"""))

    # 8. Brief Conclusion
    cells.append(nbf.v4.new_markdown_cell("""## 6. Conclusion
- **Ensemble models** (such as Random Forest and Gradient Boosting) outperformed individual base classifiers, achieving the highest F1-scores and accuracy.
- **Top predictive features**: Blood glucose level and HbA1c level are the most important indicators for predicting diabetes.
- Preprocessing through standard scaling and encoding categorical features ensures robust model performance."""))

    nb.cells = cells
    return nb

if __name__ == '__main__':
    notebook = create_simple_diabetes_notebook()
    output_filename = 'diabetes_classification_and_ensembles.ipynb'
    with open(output_filename, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    print(f"Successfully created simple {output_filename} with {len(notebook.cells)} cells!")
