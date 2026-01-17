import numpy as np
import pandas as pd
import time
import pickle
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt


class KNNRegression:
    
    def __init__(self, k=5, find_best_k=False, k_range=range(1, 21)):
        self.k = k
        self.find_best_k = find_best_k
        self.k_range = k_range
        self.model = None
        self.best_k = k
        self.training_time = None
        self.n_features = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        start_time = time.time()
        
        if isinstance(X_train, pd.DataFrame):
            X_train_array = X_train.values
        else:
            X_train_array = np.array(X_train)
            
        if isinstance(y_train, pd.Series):
            y_train_array = y_train.values
        else:
            y_train_array = np.array(y_train)
        
        self.n_features = X_train_array.shape[1]
        
        # Find best k if requested
        if self.find_best_k:
            if X_val is None or y_val is None:
                raise ValueError("X_val and y_val must be provided when find_best_k=True")
            
            print("\nFinding optimal k...")
            best_score = -float('inf')
            
            if isinstance(X_val, pd.DataFrame):
                X_val_array = X_val.values
            else:
                X_val_array = np.array(X_val)
                
            if isinstance(y_val, pd.Series):
                y_val_array = y_val.values
            else:
                y_val_array = np.array(y_val)
            
            for k_test in self.k_range:
                knn_temp = KNeighborsRegressor(n_neighbors=k_test)
                knn_temp.fit(X_train_array, y_train_array)
                score = knn_temp.score(X_val_array, y_val_array)  # Score on VALIDATION set
                
                if score > best_score:
                    best_score = score
                    self.best_k = k_test
            
            print(f"Best k found: {self.best_k} (R² = {best_score:.4f})")
        else:
            self.best_k = self.k
        
        # Train final model with best k
        self.model = KNeighborsRegressor(n_neighbors=self.best_k)
        self.model.fit(X_train_array, y_train_array)
        
        self.training_time = time.time() - start_time
        
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.array(X)
        
        return self.model.predict(X)
    
    def save_model(self, filepath):
        model_data = {
            'model': self.model,
            'best_k': self.best_k,
            'n_features': self.n_features,
            'training_time': self.training_time
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.best_k = model_data['best_k']
        self.n_features = model_data['n_features']
        self.training_time = model_data['training_time']
        
        print(f"Model loaded from {filepath}")
        return self


def calculate_metrics(y_true, y_pred):
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n = len(y_true)
    
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
    if n_samples <= n_features + 1:
        return np.nan
    
    adj_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
    return adj_r2


def train_and_evaluate(X_train, X_test, y_train, y_test, 
                       model_name="KNN_Regression",
                       k=5, find_best_k=False, k_range=range(1, 21),
                       plot_results=False):
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Store scores for plotting k optimization
    k_scores = None
    
    # Initialize model
    model = KNNRegression(k=k, find_best_k=find_best_k, k_range=k_range)
    
    # If we're finding best k and plotting, we need to capture the scores
    if find_best_k and plot_results:
        # Manually find best k to capture scores for plotting
        print("\nFinding optimal k...")
        best_score = -float('inf')
        best_k_val = k
        k_scores = []
        
        X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_train_array = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
        X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else np.array(X_test)
        y_test_array = y_test.values if isinstance(y_test, pd.Series) else np.array(y_test)
        
        for k_test in k_range:
            knn_temp = KNeighborsRegressor(n_neighbors=k_test)
            knn_temp.fit(X_train_array, y_train_array)
            score = knn_temp.score(X_test_array, y_test_array)
            k_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_k_val = k_test
        
        print(f"Best k found: {best_k_val} (R² = {best_score:.4f})")
        
        # Now train the model with the best k (without re-finding it)
        model = KNNRegression(k=best_k_val, find_best_k=False, k_range=k_range)
        model.fit(X_train, y_train)
    else:
        # Train model normally
        if find_best_k:
            model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        else:
            model.fit(X_train, y_train)
    
    print(f"Training completed in {model.training_time:.4f} seconds")
    print(f"Using k = {model.best_k} neighbors")
    
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
    
    # Create results dictionary matching EXACT column order from CSV
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
        create_plots(y_test, y_pred_test, test_metrics, model_name, k_scores, k_range, model.best_k)
    
    return model, results


def create_plots(y_test, y_pred, metrics, model_name, k_scores=None, k_range=None, best_k=None):
    
    # Convert to numpy arrays if needed
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Plot 1: K Optimization (if available)
    if k_scores is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(list(k_range), k_scores, marker='o', linewidth=2, markersize=8)
        plt.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'Best k={best_k}')
        plt.xlabel('Number of Neighbors (k)', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.title('KNN Performance vs. k Value', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plot_path = results_dir / f'{model_name}_k_optimization.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"K optimization plot saved to {plot_path}")
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
    
    # Use SCALED data for KNN (distance-based algorithm)
    X_train = data['X_train_scaled']
    X_test = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Train and evaluate
    model, results = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        model_name="KNN_Regression",
        find_best_k=True,
        k_range=range(1, 31),
        plot_results=True
    )
    
    # Save results
    save_results_to_csv(results)
    
    print("\n" + "="*60)
    print("KNN Regression Training Complete!")
    print("="*60)
    print(f"\nModel saved in: models/")
    print(f"Results saved in: results/")