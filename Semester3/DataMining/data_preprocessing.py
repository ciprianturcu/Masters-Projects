
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(filepath):
    """Load the dataset from CSV file."""
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values_mean(df) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    """
    missing_count = df.isnull().sum().sum()
    
    if missing_count == 0:
        print("No missing values found")
        return df
    
    df = df.copy()
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df


def handle_outliers(df, threshold=1.5) -> pd.DataFrame:
    """
    Handle outliers by capping them.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    outliers_count = 0
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df[col] = np.clip(df[col], lower, upper)
            outliers_count += outliers
    return df


def encode_categorical(df) -> pd.DataFrame:
    """
    Encode categorical variables using Label Encoding.
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) == 0:
        print("No categorical variables to encode")
        return df
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df


def normalize_column_names(df):
    """Clean up column names (remove spaces, special characters)."""
    df = df.copy()
    df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('/', '_')
    return df


def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Parameters:
    -----------
    X_train, X_test : DataFrame
        Train and test features
    
    Returns:
    --------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_gym_data(filepath):
    df = load_data(filepath)
    
    df = handle_missing_values_mean(df)
    
    df = handle_outliers(df)
    
    df = normalize_column_names(df)
    
    df = encode_categorical(df)
    
    X_train, X_test, y_train, y_test = split_data(df, "Calories_Burned")

    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
    }


if __name__ == "__main__":
    
    data = preprocess_gym_data('gym_members_exercise_tracking.csv')
    
    
    # Example: Linear Regression
    print("\n2. Linear Regression (use scaled data):")
    print(f"   X_train_scaled shape: {data['X_train_scaled'].shape}")
    print(f"   X_test_scaled shape: {data['X_test_scaled'].shape}")

    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression()
    # model.fit(data['X_train_scaled'], data['y_train'])
    # predictions = model.predict(data['X_test_scaled'])
    
    # Example: Random Forest
    print("\n3. Random Forest (use unscaled data):")
    print(f"   X_train shape: {data['X_train'].shape}")
    print(f"   X_test shape: {data['X_test'].shape}")

    # from sklearn.ensemble import RandomForestRegressor
    # model = RandomForestRegressor()
    # model.fit(data['X_train'], data['y_train'])
    # predictions = model.predict(data['X_test'])