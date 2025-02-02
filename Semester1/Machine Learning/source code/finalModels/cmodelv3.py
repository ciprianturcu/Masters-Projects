import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import sem, t, norm


# Sigmoid function and its derivative
# Sigmoid function maps input values to the range (0, 1), crucial for logistic regression and calculating probabilities.
def sig1(z):
    return 1 / (1 + np.exp(-z))

# Derivative of the sigmoid function, used for gradient calculations.
def sig2(z):
    phat = sig1(z)  # Compute sigmoid
    return phat * (1 - phat)  # Derivative formula

# Safe logarithm function to prevent numerical issues with log(0)
def safe_log(x, eps=1e-15):
    """Applies log safely to avoid log(0)."""
    return np.log(np.maximum(x, eps))  # Ensure input to log is never less than a small positive value

# Class to convert ordinal labels into numerical levels for processing
class y2ord():
    def __init__(self):
        self.di = {}  # Initialize dictionary for label to level mapping

    def fit(self, y):
        self.uy = np.sort(np.unique(y))  # Sort unique labels
        self.di = dict(zip(self.uy, np.arange(len(self.uy)) + 1))  # Map labels to 1, 2, ... K

    def transform(self, y):
        return np.array([self.di[z] for z in y])  # Convert each label to its corresponding level

# Conversion functions between alpha and theta parameters
# function to allow transition between the unconstrained alpha space and the constrained theta space (cumulative thresholds).
def alpha2theta(alpha, K):
    return np.cumsum(np.append(alpha[0], np.exp(alpha[1:])))  # Compute theta cumulatively from alpha

def theta2alpha(theta, K):
    return np.append(theta[0], np.log(theta[1:] - theta[:-1]))  # Reverse calculation to find alpha

# Wrapper to unpack alpha and beta parameters from a combined parameter vector
def alpha_beta_wrapper(alpha_beta, X, lb=20, ub=20):
    K = len(alpha_beta) + 1  # Total number of ordinal levels
    if X is not None:
        K -= X.shape[1]
        beta = alpha_beta[K - 1:]  # Extract beta parameters for features
    else:
        beta = np.array([0])  # Default beta to zero if no features
    alpha = alpha_beta[:K - 1]  # Extract alpha parameters for thresholds
    theta = alpha2theta(alpha, K)  # Convert alpha to theta (thresholds)
    theta = np.append(np.append(theta[0] - lb, theta), theta[-1] + ub)  # Add bounds to theta
    return alpha, theta, beta, K  # Return unpacked parameters

# Negative log-likelihood function for ordinal regression
# function is minimized to train the model.
def nll_ordinal(alpha_beta, X, idx_y, lb=20, ub=20):
    alpha, theta, beta, K = alpha_beta_wrapper(alpha_beta, X, lb, ub)  # Unpack parameters
    score = np.dot(X, beta)  # Linear predictor
    ll = 0  # Initialize log-likelihood
    for kk, idx in enumerate(idx_y):  # Iterate over ordinal levels
        ll += sum(safe_log(sig1(theta[kk + 1] - score[idx]) - sig1(theta[kk] - score[idx])))  # Log-probabilities
    nll = -1 * (ll / X.shape[0])  # Average negative log-likelihood
    return nll

# Gradient functions for optimization
# compute gradients needed for minimizing the negative log-likelihood.
def gll_ordinal(alpha_beta, X, idx_y, lb=20, ub=20):
    grad_alpha = gll_alpha(alpha_beta, X, idx_y, lb, ub)  # Gradient wrt alpha
    grad_X = gll_beta(alpha_beta, X, idx_y, lb, ub)  # Gradient wrt beta
    return np.append(grad_alpha, grad_X)  # Combine gradients

# Gradient wrt beta (feature weights)
def gll_beta(alpha_beta, X, idx_y, lb=20, ub=20):
    alpha, theta, beta, K = alpha_beta_wrapper(alpha_beta, X, lb, ub)  # Unpack parameters
    score = np.dot(X, beta)  # Linear predictor
    grad_X = np.zeros(X.shape[1])  # Initialize gradient
    for kk, idx in enumerate(idx_y):  # Iterate over ordinal levels
        den = sig1(theta[kk + 1] - score[idx]) - sig1(theta[kk] - score[idx])  # Denominator
        num = -sig2(theta[kk + 1] - score[idx]) + sig2(theta[kk] - score[idx])  # Numerator
        grad_X += np.dot(X[idx].T, num / np.maximum(den, 1e-15))  # Update gradient
    grad_X = -1 * grad_X / X.shape[0]  # Average negative gradient
    return grad_X

# Gradient wrt alpha (thresholds)
def gll_alpha(alpha_beta, X, idx_y, lb=20, ub=20):
    alpha, theta, beta, K = alpha_beta_wrapper(alpha_beta, X, lb, ub)  # Unpack parameters
    score = np.dot(X, beta)  # Linear predictor
    grad_alpha = np.zeros(K - 1)  # Initialize gradient
    for kk in range(K - 1):  # Iterate over thresholds
        idx_p, idx_n = idx_y[kk], idx_y[kk + 1]  # Indices for current and next level
        den_p = sig1(theta[kk + 1] - score[idx_p]) - sig1(theta[kk] - score[idx_p])  # Denominator for positive class
        den_n = sig1(theta[kk + 2] - score[idx_n]) - sig1(theta[kk + 1] - score[idx_n])  # Denominator for negative class
        num_p, num_n = sig2(theta[kk + 1] - score[idx_p]), sig2(theta[kk + 1] - score[idx_n])  # Numerators
        grad_alpha[kk] += sum(num_p / np.maximum(den_p, 1e-15)) - sum(num_n / np.maximum(den_n, 1e-15))  # Gradient update
    grad_alpha = -1 * grad_alpha / X.shape[0]  # Average negative gradient
    grad_alpha *= np.append(1, np.exp(alpha[1:]))  # Chain rule adjustment
    return grad_alpha

# Probability inference function
# Compute probabilities for each ordinal class given model parameters.
def prob_ordinal(alpha_beta, X, lb=20, ub=20):
    alpha, theta, beta, K = alpha_beta_wrapper(alpha_beta, X, lb, ub)  # Unpack parameters
    score = np.dot(X, beta)  # Linear predictor
    phat = (np.atleast_2d(theta) - np.atleast_2d(score).T)  # Calculate thresholds relative to score
    phat = sig1(phat[:, 1:]) - sig1(phat[:, :-1])  # Compute probabilities
    return phat

# Encapsulates training, prediction, and probability estimation for ordinal regression.
class ordinal_reg():
    def __init__(self, standardize=True, lb=20, ub=20):
        self.standardize = standardize  # Whether to standardize features
        self.lb = lb  # Lower bound for thresholds
        self.ub = ub  # Upper bound for thresholds

    def fit(self, data, lbls):
        self.p = data.shape[1]  # Number of features
        self.Xenc = StandardScaler().fit(data)  # Standardize features
        self.yenc = y2ord()  # Initialize ordinal encoder
        self.yenc.fit(y=lbls)  # Fit ordinal encoder
        ytil = self.yenc.transform(lbls)  # Transform labels
        idx_y = [np.where(ytil == yy)[0] for yy in list(self.yenc.di.values())]  # Group indices by class
        self.K = len(idx_y)  # Number of classes
        theta_init = np.array([(z + 1) / self.K for z in range(self.K - 1)])  # Initial thresholds
        theta_init = np.log(theta_init / (1 - theta_init))  # Logit transform
        alpha_init = theta2alpha(theta_init, self.K)  # Convert thresholds to alpha
        param_init = np.append(alpha_init, np.repeat(0, self.p))  # Combine alpha and beta
        self.alpha_beta = minimize(fun=nll_ordinal, x0=param_init, method='L-BFGS-B', jac=gll_ordinal,
                                   args=(self.Xenc.transform(data), idx_y, self.lb, self.ub)).x  # Optimize parameters

    def predict(self, data):
        phat = prob_ordinal(self.alpha_beta, self.Xenc.transform(data), self.lb, self.ub)  # Predict probabilities
        return np.argmax(phat, axis=1) + 1  # Return class with highest probability

    def predict_proba(self, data):
        phat = prob_ordinal(self.alpha_beta, self.Xenc.transform(data), self.lb, self.ub)  # Predict probabilities
        return phat


file_path = '../dataset/gym_members_exercise_tracking.csv'
data = pd.read_csv(file_path)
original_data = data.copy()


features = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
            'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
            'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days per week)', 'BMI']  # Define feature columns


target = 'Experience_Level'
data[target] = pd.Categorical(data[target], ordered=True)


data = pd.get_dummies(data, columns=['Gender', 'Workout_Type'], drop_first=True)

# Prepare data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_copy = X_train.copy()
y_train_copy = y_train.copy()
X_test_copy = X_test.copy()
y_test_copy = y_test.copy()

# Convert to numpy arrays for cross-validation
X_array = X.to_numpy()
y_array = y.cat.codes.to_numpy()

# Perform hyperparameter optimization
param_grid = {
    'lb': [5,15, 20, 25],  # Lower bound for thresholds
    'ub': [5,15, 20, 25],  # Upper bound for thresholds
    'standardize': [True, False]  # Whether to standardize features
}

grid = ParameterGrid(param_grid)

best_params = None
best_score = -np.inf

# Store all accuracies for each combination
all_accuracies = []

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for params in grid:
    lb = params['lb']
    ub = params['ub']
    standardize = params['standardize']

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_aucs = []
    fold_auprcs = []

    for train_index, test_index in kf.split(X_array):
        X_train, X_test = X_array[train_index], X_array[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = ordinal_reg(standardize=standardize, lb=lb, ub=ub)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # AUC and AUPRC
        try:
            y_proba = model.predict_proba(X_test)
            auc = roc_auc_score(pd.get_dummies(y_test), y_proba, multi_class='ovr')
            auprc = average_precision_score(pd.get_dummies(y_test), y_proba, average='weighted')
        except ValueError:
            auc = None
            auprc = None

        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)
        if auc is not None:
            fold_aucs.append(auc)
        if auprc is not None:
            fold_auprcs.append(auprc)

    avg_accuracy = np.mean(fold_accuracies)
    avg_precision = np.mean(fold_precisions)
    avg_recall = np.mean(fold_recalls)
    avg_f1 = np.mean(fold_f1s)
    avg_auc = np.mean(fold_aucs) if fold_aucs else None
    avg_auprc = np.mean(fold_auprcs) if fold_auprcs else None

    #standard deviation and confidence intervals
    accuracy_std = np.std(fold_accuracies)
    confidence = 0.95
    n_folds = len(fold_accuracies)
    h = sem(fold_accuracies) * t.ppf((1 + confidence) / 2, n_folds - 1)

    all_accuracies.append({
        'lb': lb,
        'ub': ub,
        'standardize': standardize,
        'accuracy': avg_accuracy,
        'confidence_interval': (avg_accuracy - h, avg_accuracy + h),
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1,
        'auc': avg_auc,
        'auprc': avg_auprc
    })

    if avg_accuracy > best_score:
        best_score = avg_accuracy
        best_params = params

print("All Hyperparameter Combinations and Accuracies:")
for result in all_accuracies:
    print(result)

print("Best Hyperparameters:", best_params)
print("Best Cross-Validated Accuracy:", best_score)

final_model = ordinal_reg(standardize=best_params['standardize'], lb=best_params['lb'], ub=best_params['ub'])
final_model.fit(X_train_copy, y_train_copy)

final_predictions = final_model.predict(X_test_copy)
accuracy = accuracy_score(y_test_copy, final_predictions)
precision = precision_score(y_test_copy, final_predictions, average='weighted')
recall = recall_score(y_test_copy, final_predictions, average='weighted')
f1 = f1_score(y_test_copy, final_predictions, average='weighted')
conf_matrix = confusion_matrix(y_test_copy, final_predictions)

# AUC and AUPRC
try:
    y_proba = final_model.predict_proba(X)
    auc = roc_auc_score(pd.get_dummies(y), y_proba, multi_class='ovr')
    auprc = average_precision_score(pd.get_dummies(y), y_proba, average='weighted')
except ValueError:
    auc = None
    auprc = None

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
if auc is not None:
    print(f"AUC: {auc:.4f}")
if auprc is not None:
    print(f"AUPRC: {auprc:.4f}")
n = len(y_test)
z = norm.ppf(0.975)  # 95% confidence interval
accuracy_ci = z * np.sqrt((accuracy * (1 - accuracy)) / n)
print(accuracy_ci)
print('Confusion Matrix:')
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Beginner", "Intermediate", "Expert"], yticklabels=["Beginner", "Intermediate", "Expert"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusionMatrixScratch.png')
plt.show()

if isinstance(X_train, np.ndarray):
    X_train_df = pd.DataFrame(X_train, columns=features)
    X_test_df = pd.DataFrame(X_test, columns=features)
else:
    X_train_df = X_train
    X_test_df = X_test

# Initialize LIME Tabular Explainer
explainer = LimeTabularExplainer(
    training_data=X_train_copy.values,
    training_labels=y_train.cat.codes,
    feature_names=features,
    class_names=list(data[target].cat.categories),
    mode='classification',
    discretize_continuous=False
)

instance_idx = 0
instance_to_explain = X_test_copy.iloc[instance_idx].values

lime_explanation = explainer.explain_instance(
    data_row=instance_to_explain,
    predict_fn=final_model.predict_proba  
)

lime_explanation.as_pyplot_figure()
plt.tight_layout()
plt.title(f'LIME Explanation for Test Instance {instance_idx}')
plt.savefig("limeExplanationScratch.png")
plt.show()
