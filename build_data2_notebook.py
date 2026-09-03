import nbformat as nbf

def generate_simple_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Imports & Load Data
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('data (2).csv')
df.head()"""))

    # Cell 2: Preprocessing
    cells.append(nbf.v4.new_code_cell("""# Data Preprocessing
X = pd.get_dummies(df.drop('loan_status', axis=1), drop_first=True)
y = df['loan_status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)"""))

    # Cell 3: Decision Tree
    cells.append(nbf.v4.new_code_cell("""# 1. Decision Tree Model
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_dt))"""))

    # Cell 4: Naive Bayes
    cells.append(nbf.v4.new_code_cell("""# 2. Naive Bayes Model
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

y_pred_nb = nb.predict(X_test_scaled)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_nb))"""))

    # Cell 5: SVM
    cells.append(nbf.v4.new_code_cell("""# 3. Support Vector Machine (SVM) Model
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_svm))"""))

    # Cell 6: Compare Models
    cells.append(nbf.v4.new_code_cell("""# 4. Compare Models
results = pd.DataFrame({
    'Model': ['Decision Tree', 'Naive Bayes', 'SVM'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_svm)
    ]
})
print(results)

# Comparison Graph
plt.figure(figsize=(6, 4))
sns.barplot(x='Model', y='Accuracy', data=results)
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.show()"""))

    nb.cells = cells
    return nb

if __name__ == '__main__':
    nb = generate_simple_notebook()
    with open('data2.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Ultra-simple data2.ipynb created successfully!")
