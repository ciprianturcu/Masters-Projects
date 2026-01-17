# Calories Burned Prediction - Machine Learning Implementation

## Contributions

My contribution to this project includes the **Linear Regression model implementation** and **dataset search/selection**.

## Project Structure

```
Master-DataMining/
├── data_preprocessing.py
├── linear_regression.py
├── knn_regression.py
├── gradient_boosting.py
├── gym_members_exercise_tracking.csv
├── models/
│   └── Linear_Regression.pkl
│   └── KNN_Regression.pkl
│   └── Gradient_Boosting.pkl
├── results/
│   └── model_metrics.csv
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
   - **Windows**: `.venv\Scripts\activate`
   - **macOS/Linux**: `source .venv/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Linear Regression Training
```bash
python linear_regression.py
```

### Run KNN Regression Training
```bash
python knn_regression.py
```

### Run Gradient Boosting Training
```bash
python gradient_boosting.py
```

Each script will:
- Load and preprocess the gym dataset
- Train the respective model
- Evaluate on both training and test sets
- Calculate all required metrics
- Save the trained model to `models/`
- Export results to CSV files in `results/`

---

## Implementation Details

### 1. Linear Regression

The implementation uses the **Normal Equation** method to find optimal weights:

```
θ = (X^T X)^(-1) X^T y
```

Where:
- θ = model parameters (weights + bias)
- X = feature matrix
- y = target values

**Why Normal Equation?**

1. **Exact Solution**: Computes the optimal weights directly (no iterations)
2. **Fast**: For datasets with moderate features (< 10,000), it's very efficient
3. **No Hyperparameters**: No learning rate or iterations to tune
4. **Stable**: Guaranteed to converge to the global minimum

---

### 2. KNN Regression (K-Nearest Neighbors)

The implementation uses a **distance-based approach** to predict values:

```
ŷ = (1/k) * Σ y_i  for i in k nearest neighbors
```

Where:
- ŷ = predicted value
- k = number of neighbors
- y_i = target values of nearest neighbors

**How KNN Regression Works:**

1. **Distance Calculation**: For a new data point, calculate the distance (Euclidean) to all training points
2. **Find Neighbors**: Select the k closest training points
3. **Average**: Predict the average of the target values of those k neighbors

**Why Use Scaled Data for KNN?**

KNN is distance-based, so features with larger scales would dominate the distance calculation. StandardScaler ensures all features contribute equally.

**Hyperparameters:**
- `k`: Number of neighbors (default: 5, can be optimized)
- `k_range`: Range to search for optimal k when `find_best_k=True`

---

### 3. Gradient Boosting Regression

The implementation builds an **ensemble of decision trees** sequentially:

```
F_m(x) = F_{m-1}(x) + η * h_m(x)
```

Where:
- F_m(x) = prediction at iteration m
- η = learning rate
- h_m(x) = weak learner (decision tree) fitted to residuals

**How Gradient Boosting Works:**

1. **Initialize**: Start with a constant prediction (mean of target values)
2. **Compute Residuals**: Calculate errors (y - current_prediction)
3. **Fit Tree**: Train a decision tree to predict the residuals
4. **Update**: Add the tree's predictions (scaled by learning rate) to the model
5. **Repeat**: Continue for n_estimators iterations

**Key Concepts:**
- **Loss Function**: MSE = (1/2) * Σ(y - F(x))²
- **Negative Gradient**: For MSE, this equals the residuals (y - F(x))
- **Weak Learners**: Shallow decision trees (stumps) that capture patterns in residuals

**Why Use Unscaled Data for Gradient Boosting?**

Tree-based methods split on feature values directly and are invariant to monotonic transformations. Scaling doesn't affect the splits or predictions.

**Hyperparameters:**
- `n_estimators`: Number of trees (default: 100)
- `learning_rate`: Step size for updates (default: 0.1)
- `max_depth`: Maximum depth of each tree (default: 3)
- `subsample`: Fraction of samples used per tree (default: 0.8)

---

## Evaluation Metrics

All models are evaluated using the following metrics:

1. **RMSE (Root Mean Squared Error)**
   - Penalizes larger errors more heavily
   - Interpretable in original units (calories)
   - Formula: `sqrt(mean((y_true - y_pred)^2))`

2. **MAE (Mean Absolute Error)**
   - Robust to outliers
   - Easier to interpret as average prediction error
   - Formula: `mean(|y_true - y_pred|)`

3. **R² (R-squared)**
   - Explains variance captured by the model
   - Range: 0 to 1 (1 = perfect fit)
   - Formula: `1 - (SS_res / SS_tot)`

4. **MAPE (Mean Absolute Percentage Error)**
   - Shows relative error as percentage
   - Useful for understanding prediction quality
   - Formula: `mean(|y_true - y_pred| / y_true) * 100`

5. **Adjusted R²**
   - Accounts for number of features
   - Better for comparing models with different complexities
   - Formula: `1 - [(1 - R²) * (n - 1) / (n - p - 1)]`

## License

MIT License - Data Mining Final Project