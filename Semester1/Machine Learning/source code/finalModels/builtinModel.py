import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from statsmodels.miscmodels.ordinal_model import OrderedModel

def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.savefig('confusionMatrixBuiltIn.png', dpi=300)
    plt.show()

data = pd.read_csv('../dataset/gym_members_exercise_tracking.csv')

target = 'Experience_Level'

X = data.drop(columns=[target])
y = data[target]

label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Map Experience_Level to integers 1, 2, 3
experience_mapping = {1: 0, 2: 1, 3: 2}
y = y.map(experience_mapping)
y = pd.Categorical(y, categories=[0, 1, 2], ordered=True)

# Perform K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
methods = ['bfgs', 'newton', 'lbfgs']
best_score = -np.inf
best_method = None

for method in methods:
    fold_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        try:
            model = OrderedModel(y_train, X_train, distr='logit')
            result = model.fit(method=method, disp=False)

            pred_probs = result.predict(X_val)
            y_pred = np.argmax(pred_probs, axis=1)
            fold_scores.append(accuracy_score(y_val.codes, y_pred))
        except Exception as e:
            print(f"Error with method '{method}': {e}")

    mean_score = np.mean(fold_scores)

    if mean_score > best_score:
        best_score = mean_score
        best_method = method

print(f"Best Method: {best_method} with mean accuracy: {best_score:.4f}")

# Fit the best model on the full training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
final_model = OrderedModel(y_train, X_train, distr='logit')
final_result = final_model.fit(method=best_method, disp=False)


pred_probs = final_result.predict(X_test)
y_pred = np.argmax(pred_probs, axis=1)


reverse_experience_mapping = {0: 1, 1: 2, 2: 3}
y_test_original = y_test.map(reverse_experience_mapping)
y_pred_original = pd.Series(y_pred).map(reverse_experience_mapping)


accuracy = accuracy_score(y_test.codes, y_pred)
precision = precision_score(y_test.codes, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test.codes, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test.codes, y_pred, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(y_test.codes, y_pred)


print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print('Confusion Matrix:')
n = len(y_test)
z = norm.ppf(0.975)  # 95% confidence interval
accuracy_ci = z * np.sqrt((accuracy * (1 - accuracy)) / n)
print(accuracy_ci)
print(conf_matrix)
plot_confusion_matrix(conf_matrix, class_names=["Beginner", "Intermediate", "Expert"])


explainer = LimeTabularExplainer(
    training_data=X_train.values,
    training_labels=y_train.codes,
    feature_names=X_train.columns.tolist(),
    class_names=["Beginner", "Intermediate", "Expert"],
    mode='classification',
    discretize_continuous=False
)


instance_idx = 0
instance_to_explain = X_test.iloc[instance_idx].values


lime_explanation = explainer.explain_instance(
    data_row=instance_to_explain,
    predict_fn=lambda x: np.column_stack([
        1 - final_result.predict(pd.DataFrame(x, columns=X_test.columns)).sum(axis=1),
        final_result.predict(pd.DataFrame(x, columns=X_test.columns))
    ])
)

lime_explanation.as_pyplot_figure()
plt.tight_layout()
plt.title(f'LIME Explanation for Test Instance {instance_idx}')
plt.savefig('limeExplanationBuiltIn.png')
plt.show()

