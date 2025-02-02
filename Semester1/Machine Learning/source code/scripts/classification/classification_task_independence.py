from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = '../../dataset/gym_members_exercise_tracking.csv'
df = pd.read_csv(file_path)


target = 'Experience_Level'
X = df.drop(columns=[target])
y = df[target]

for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

mi_scores = mutual_info_classif(X, y, discrete_features='auto')  # 'auto' detects discrete features automatically
mi_scores_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores})

mi_scores_df = mi_scores_df.sort_values(by='Mutual Information', ascending=False).reset_index(drop=True)

print(mi_scores_df)

mi_scores_df.to_csv('../../output/classification/mutual_information_scores.csv', index=False)