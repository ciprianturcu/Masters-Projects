import numpy as np
import pandas as pd
import time
import pickle
from pathlib import Path
import matplotlib.pyplot as plt


class DecisionTreeStump:
    """
    A simple decision tree with limited depth for use as weak learner.
    Implements a basic regression tree using recursive binary splitting.
    """
    
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X, y):
        """Build the decision tree."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        X = np.array(X)
        y = np.array(y)
        
        self.tree = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth):
        """Recursively build the tree."""
        n_samples, n_features = X.shape
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return {'leaf': True, 'value': np.mean(y)}
        
        # Find the best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no valid split found, return leaf
        if best_gain == 0:
            return {'leaf': True, 'value': np.mean(y)}
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        n_samples, n_features = X.shape
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        # Current variance (MSE from mean)
        current_mse = np.var(y) * n_samples
        
        for feature in range(n_features):
            # Get unique thresholds
            thresholds = np.unique(X[:, feature])
            
            # Try fewer thresholds for efficiency
            if len(thresholds) > 10:
                thresholds = np.percentile(X[:, feature], np.linspace(10, 90, 10))
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue
                
                # Calculate MSE for each split
                left_mse = np.var(y[left_mask]) * np.sum(left_mask) if np.sum(left_mask) > 0 else 0
                right_mse = np.var(y[right_mask]) * np.sum(right_mask) if np.sum(right_mask) > 0 else 0
                
                # Calculate gain
                gain = current_mse - (left_mse + right_mse)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def predict(self, X):
        """Predict values for input samples."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)
        
        predictions = np.array([self._predict_single(x, self.tree) for x in X])
        return predictions
    
    def _predict_single(self, x, tree):
        """Predict a single sample."""
        if tree['leaf']:
            return tree['value']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])


class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor implemented from scratch.
    
    Uses decision tree stumps as weak learners and optimizes
    using gradient descent on the residuals (negative gradients).
    
    For regression with MSE loss:
    - Loss function: L(y, F) = (1/2) * (y - F)^2
    - Negative gradient (pseudo-residual): r = y - F(x)
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, subsample=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        
        self.trees = []
        self.initial_prediction = None
        self.training_time = None
        self.n_features = None
        self.train_scores = []  # Track training progress
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit the gradient boosting model.
        
        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            Training features
        y_train : array-like of shape (n_samples,)
            Training target values
        X_val : array-like, optional
            Validation features for tracking progress
        y_val : array-like, optional
            Validation target values
        """
        start_time = time.time()
        np.random.seed(self.random_state)
        
        # Convert to numpy arrays
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        n_samples, self.n_features = X_train.shape
        
        # Initialize with mean prediction (optimal constant model for MSE)
        self.initial_prediction = np.mean(y_train)
        
        # Current predictions
        F = np.full(n_samples, self.initial_prediction)
        
        self.trees = []
        self.train_scores = []
        
        print(f"\nTraining Gradient Boosting with {self.n_estimators} estimators...")
        
        for i in range(self.n_estimators):
            # Compute negative gradients (pseudo-residuals)
            # For MSE loss: negative gradient = y - F(x) = residuals
            residuals = y_train - F
            
            # Subsample if specified
            if self.subsample < 1.0:
                sample_size = int(n_samples * self.subsample)
                indices = np.random.choice(n_samples, sample_size, replace=False)
                X_subset = X_train[indices]
                residuals_subset = residuals[indices]
            else:
                X_subset = X_train
                residuals_subset = residuals
            
            # Fit a weak learner to the pseudo-residuals
            tree = DecisionTreeStump(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_subset, residuals_subset)
            
            # Update predictions
            predictions = tree.predict(X_train)
            F += self.learning_rate * predictions
            
            # Store the tree
            self.trees.append(tree)
            
            # Track progress every 10 iterations
            if (i + 1) % 10 == 0 or i == 0:
                train_mse = np.mean((y_train - F) ** 2)
                train_r2 = 1 - (np.sum((y_train - F) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2))
                self.train_scores.append({'iteration': i + 1, 'mse': train_mse, 'r2': train_r2})
                
                if (i + 1) % 50 == 0:
                    print(f"  Iteration {i + 1}/{self.n_estimators}: Train R² = {train_r2:.4f}")
        
        self.training_time = time.time() - start_time
        
        return self
    
    def predict(self, X):
        """
        Predict target values for input samples.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        predictions : array of shape (n_samples,)
            Predicted values
        """
        if self.initial_prediction is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)
        
        # Start with initial prediction
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        # Add contribution from each tree
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions
    
    def save_model(self, filepath):
        """Save the model to a file."""
        model_data = {
            'trees': self.trees,
            'initial_prediction': self.initial_prediction,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'n_features': self.n_features,
            'training_time': self.training_time,
            'train_scores': self.train_scores
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the model from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.trees = model_data['trees']
        self.initial_prediction = model_data['initial_prediction']
        self.n_estimators = model_data['n_estimators']
        self.learning_rate = model_data['learning_rate']
        self.max_depth = model_data['max_depth']
        self.n_features = model_data['n_features']
        self.training_time = model_data['training_time']
        self.train_scores = model_data.get('train_scores', [])
        
        print(f"Model loaded from {filepath}")
        return self


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # MSE and RMSE
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'MSE': mse
    }


def calculate_adjusted_r2(r2, n_samples, n_features):
    """Calculate adjusted R-squared."""
    if n_samples <= n_features + 1:
        return np.nan
    
    adj_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
    return adj_r2


def train_and_evaluate(X_train, X_test, y_train, y_test,
                       model_name="Gradient_Boosting",
                       n_estimators=100, learning_rate=0.1, max_depth=3,
                       subsample=1.0, plot_results=False):
    """
    Train and evaluate the Gradient Boosting model.
    
    Parameters:
    -----------
    X_train, X_test : DataFrame or array
        Training and test features
    y_train, y_test : Series or array
        Training and test targets
    model_name : str
        Name for saving the model
    n_estimators : int
        Number of boosting iterations
    learning_rate : float
        Learning rate (shrinkage)
    max_depth : int
        Maximum depth of each tree
    subsample : float
        Fraction of samples to use for each tree
    plot_results : bool
        Whether to create visualization plots
        
    Returns:
    --------
    model : GradientBoostingRegressor
        Trained model
    results : dict
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"\nHyperparameters:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  max_depth: {max_depth}")
    print(f"  subsample: {subsample}")
    
    # Initialize and train model
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample
    )
    
    model.fit(X_train, y_train)
    
    print(f"\nTraining completed in {model.training_time:.4f} seconds")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    # Calculate adjusted R²
    n_samples = len(y_test)
    n_features = X_test.shape[1]
    adjusted_r2 = calculate_adjusted_r2(test_metrics['R2'], n_samples, n_features)
    
    # Print results
    print(f"\nTraining Set Performance:")
    print(f"  RMSE: {train_metrics['RMSE']:.4f}")
    print(f"  MAE:  {train_metrics['MAE']:.4f}")
    print(f"  R²:   {train_metrics['R2']:.4f}")
    print(f"  MAPE: {train_metrics['MAPE']:.4f}%")
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE:         {test_metrics['RMSE']:.4f}")
    print(f"  MAE:          {test_metrics['MAE']:.4f}")
    print(f"  R²:           {test_metrics['R2']:.4f}")
    print(f"  MAPE:         {test_metrics['MAPE']:.4f}%")
    print(f"  Adjusted R²:  {adjusted_r2:.4f}")
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"{model_name}.pkl"
    model.save_model(model_path)
    
    # Create results dictionary
    results = {
        'Model': model_name,
        'Training_Time_seconds': model.training_time,
        'Train_RMSE': train_metrics['RMSE'],
        'Train_MAE': train_metrics['MAE'],
        'Train_R2': train_metrics['R2'],
        'Test_RMSE': test_metrics['RMSE'],
        'Test_MAE': test_metrics['MAE'],
        'Test_R2': test_metrics['R2'],
        'Test_MAPE': test_metrics['MAPE'],
        'Test_Adjusted_R2': adjusted_r2,
        'N_Features': n_features,
        'N_Train_Samples': len(y_train),
        'N_Test_Samples': n_samples
    }
    
    # Create visualizations if requested
    if plot_results:
        create_plots(y_test, y_pred_test, test_metrics, model_name, model.train_scores)
    
    return model, results


def create_plots(y_test, y_pred, metrics, model_name, train_scores=None):
    """Create visualization plots for model evaluation."""
    
    # Convert to numpy arrays if needed
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Plot 1: Training Progress
    if train_scores and len(train_scores) > 0:
        plt.figure(figsize=(10, 6))
        iterations = [s['iteration'] for s in train_scores]
        r2_scores = [s['r2'] for s in train_scores]
        plt.plot(iterations, r2_scores, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Estimators', fontsize=12)
        plt.ylabel('Training R² Score', fontsize=12)
        plt.title('Gradient Boosting Training Progress', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = results_dir / f'{model_name}_training_progress.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training progress plot saved to {plot_path}")
        plt.show()
    
    # Plot 2: Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Calories Burned', fontsize=12)
    plt.ylabel('Predicted Calories Burned', fontsize=12)
    plt.title(f'Actual vs Predicted (R²={metrics["R2"]:.4f})', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = results_dir / f'{model_name}_actual_vs_predicted.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Actual vs predicted plot saved to {plot_path}")
    plt.show()
    
    # Plot 3: Residuals
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Calories Burned', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title(f'Residual Plot (MAE={metrics["MAE"]:.2f})', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = results_dir / f'{model_name}_residuals.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Residual plot saved to {plot_path}")
    plt.show()
    
    # Plot 4: Residuals Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Residuals', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = results_dir / f'{model_name}_residuals_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Residuals distribution plot saved to {plot_path}")
    plt.show()
    
    print(f"\nAll plots saved to {results_dir}/")


def save_results_to_csv(results, metrics_filename="model_metrics.csv"):
    """Save results to CSV file, appending to existing data if file exists."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    if isinstance(results, dict):
        results = [results]
    
    # Define exact column order to match existing CSV
    column_order = [
        'Model', 'Training_Time_seconds', 'Train_RMSE', 'Train_MAE', 'Train_R2',
        'Test_RMSE', 'Test_MAE', 'Test_R2', 'Test_MAPE', 'Test_Adjusted_R2',
        'N_Features', 'N_Train_Samples', 'N_Test_Samples'
    ]
    
    df_new = pd.DataFrame(results)
    df_new = df_new[column_order]
    
    metrics_path = results_dir / metrics_filename
    
    # Check if file exists and append, otherwise create new
    if metrics_path.exists():
        df_existing = pd.read_csv(metrics_path)
        # Append new results
        df_metrics = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"\nMetrics appended to existing file: {metrics_path}")
    else:
        df_metrics = df_new
        print(f"\nMetrics saved to new file: {metrics_path}")
    
    # Save combined data
    df_metrics.to_csv(metrics_path, index=False)
    
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    
    print("\nAll Models Evaluation Metrics:")
    metrics_cols = ['Model', 'Test_RMSE', 'Test_MAE', 'Test_R2', 'Test_MAPE', 'Test_Adjusted_R2']
    print(df_metrics[metrics_cols].to_string(index=False))
    
    print("\nTraining Information:")
    time_cols = ['Model', 'Training_Time_seconds']
    print(df_metrics[time_cols].to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    from data_preprocessing import preprocess_gym_data
    
    print("Loading and preprocessing data...")
    data = preprocess_gym_data('gym_members_exercise_tracking.csv')
    
    # Use UNSCALED data for Gradient Boosting (tree-based algorithm)
    # Tree-based methods don't require feature scaling
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Train and evaluate
    model, results = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        model_name="Gradient_Boosting",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        plot_results=True
    )
    
    # Save results
    save_results_to_csv(results)
    
    print("\n" + "="*60)
    print("Gradient Boosting Training Complete!")
    print("="*60)
    print(f"\nModel saved in: models/")
    print(f"Results saved in: results/")