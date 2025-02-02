import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import dask.dataframe as dd

def print_missing_values_per_column():
    print("Missing values per column:")
    print(df.isnull().sum())

file_path = '../dataset/gym_members_exercise_tracking.csv'
df = pd.read_csv(file_path)

print_missing_values_per_column()

print("Starting to label encode categorical features")
categorical_columns = ['Gender', 'Workout_Type']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
print("Finished encoding categorical features")


correlation_matrix = dd.from_pandas(df, npartitions=10).corr().compute()


plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("../output/correlation/correlation_heatmap.png", dpi=300)
plt.show()
