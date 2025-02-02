import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = '../../dataset/gym_members_exercise_tracking.csv'
data = pd.read_csv(file_path)

# Prepare features (X) and target (y)
X = data.drop(columns=["Experience_Level"])
y = data["Experience_Level"]

# Encode categorical features
categorical_columns = X.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

# Compute feature importances
importances = random_forest.feature_importances_
feature_names = X.columns

# Sort features by importance
indices = np.argsort(importances)[::-1]
sorted_features = feature_names[indices]
sorted_importances = importances[indices]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_importances)), sorted_importances, align="center")
plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
plt.title("Feature Importance (Random Forest Classifier)")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.tight_layout()

plt.savefig("../../output/classification/feature_importance_plot.png", dpi=300)

plt.show()

feature_importance_table = pd.DataFrame({
    "Feature": sorted_features,
    "Importance": sorted_importances
})

print(feature_importance_table)

feature_importance_table.to_csv("../../output/classification/feature_importance_results.csv", index=False)
