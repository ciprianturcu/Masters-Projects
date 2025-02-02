import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    average_precision_score, roc_auc_score
import seaborn as sns
from scipy.stats import norm

data = pd.read_csv('../dataset/gym_members_exercise_tracking.csv')

features = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
            'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
            'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days per week)', 'BMI']
target = 'Experience_Level'
X = data[features]
y = data[target]


label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

class OrdinalLogisticRegression:
    def __init__(self):
        self.models = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models = []
        for k in range(len(self.classes_) - 1):
            y_bin = (y > self.classes_[k]).astype(int)
            model = LogisticRegression(max_iter=5000)
            model.fit(X, y_bin)
            self.models.append(model)
        return self

    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], len(self.classes_)))
        for k, model in enumerate(self.models):
            probas[:, k + 1] = model.predict_proba(X)[:, 1]
        probas[:, 0] = 1 - probas[:, 1]
        for k in range(1, len(self.classes_) - 1):
            probas[:, k] -= probas[:, k + 1]
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

def cross_validate_and_tune(X, y):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear']
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_params = None
    best_score = 0
    cv_accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        model = OrdinalLogisticRegression()

        for C in param_grid['C']:
            for solver in param_grid['solver']:
                try:
                    temp_model = LogisticRegression(C=C, solver=solver, max_iter=5000)
                    temp_model.fit(X_train, (y_train > 0).astype(int))  # Testing one binary threshold
                    score = temp_model.score(X_val, (y_val > 0).astype(int))
                    cv_accuracies.append(score)
                    if score > best_score:
                        best_score = score
                        best_params = {'C': C, 'solver': solver}
                except Exception as e:
                    continue

    mean_cv_accuracy = np.mean(cv_accuracies)
    print(f"Mean Cross-Validation Accuracy: {mean_cv_accuracy:.4f}")
    return best_params

best_params = cross_validate_and_tune(X, y)
print(f"Best Hyperparameters: {best_params}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_copy = X_train.copy()
y_train_copy = y_train.copy()
X_test_copy = X_test.copy()
y_test_copy = y_test.copy()

model = OrdinalLogisticRegression()
if best_params:
    LogisticRegression(C=best_params['C'], solver=best_params['solver'], max_iter=5000)
model.fit(X_train, y_train)


# Convert datasets to DataFrame for LIME compatibility
X_train_df = pd.DataFrame(X_train_copy, columns=features)
X_test_df = pd.DataFrame(X_test_copy, columns=features)

lime_explainer = LimeTabularExplainer(
    training_data=X_train_df.values,  # Use training data as NumPy array
    training_labels=y_train,  # Labels for training data
    feature_names=features,  # Feature names
    class_names=list(target_encoder.classes_),  # Class names
    mode='classification',
    discretize_continuous=False  # Disable binning for continuous features
)


instance_idx = 0
instance_to_explain = X_test_df.iloc[instance_idx].values

lime_explanation = lime_explainer.explain_instance(
    data_row=instance_to_explain,
    predict_fn=lambda x: model.predict_proba(scaler.transform(x))  # Scale input before predict_proba
)

# Visualize the explanation as a static plot
lime_explanation.as_pyplot_figure()
plt.tight_layout()
plt.title(f'LIME Explanation for Test Instance {instance_idx}')
plt.savefig('limeExplanationBuiltInV2.png')
plt.show()

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# AUC and AUPRC
try:
    y_proba = model.predict_proba(X)
    auc = roc_auc_score(pd.get_dummies(y), y_proba, multi_class='ovr')
    auprc = average_precision_score(pd.get_dummies(y), y_proba, average='weighted')
except ValueError:
    auc = None
    auprc = None

# Display the results
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
if auc is not None:
    print(f"AUC: {auc:.4f}")
if auprc is not None:
    print(f"AUPRC: {auprc:.4f}")

# Calculate confidence intervals for accuracy
n = len(y_test)
z = norm.ppf(0.975)  # 95% confidence interval
accuracy_ci = z * np.sqrt((accuracy * (1 - accuracy)) / n)
print(accuracy_ci)
print('Confusion Matrix:')
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Beginner", "Intermediate", "Expert"], yticklabels=["Beginner", "Intermediate", "Expert"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusionMatrixBuiltInV2.png')
plt.show()


