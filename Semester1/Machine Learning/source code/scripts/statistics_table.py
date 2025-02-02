import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = '../dataset/gym_members_exercise_tracking.csv'
df = pd.read_csv(file_path)

print("Starting to label encode categorical features")
categorical_columns = ['Gender', 'Workout_Type']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
print("Finished encoding categorical features")

stats_table = df.describe().T

stats_table.rename(columns={
    'count': 'Count',
    'mean': 'Mean',
    'std': 'Std Dev',
    'min': 'Min',
    '25%': '25%',
    '50%': '50%',
    '75%': '75%',
    'max': 'Max'
}, inplace=True)

output_path = '../output/statistics/descriptive_statistics.csv'
stats_table.to_csv(output_path)
print(stats_table)
