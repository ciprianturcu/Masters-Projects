import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, LabelEncoder

def print_missing_values_per_column():
    print("Missing values per column:")
    print(heart_data.isnull().sum())

heart_data = pd.read_csv('HeartDiseaseTrain-Test.csv')
print_missing_values_per_column()

categorical_cols = heart_data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    heart_data[col] = le.fit_transform(heart_data[col])

X_heart = heart_data.drop('target', axis=1)
y_heart = heart_data['target']

n_features = X_heart.shape[1]
n_classes = len(np.unique(y_heart))

n_components = min(n_features, n_classes - 1)

X_heart = StandardScaler().fit_transform(X_heart)

lda = LDA(n_components=n_components)
X_r2 = lda.fit(X_heart, y_heart).transform(X_heart)

lda_df = pd.DataFrame({'LD1': X_r2[:, 0], 'Target': y_heart})

plt.figure(figsize=(10, 6))
sns.stripplot(x='Target', y='LD1', data=lda_df, jitter=True, palette='Set1', alpha=0.7)
plt.title('LDA of Heart Disease Dataset')
plt.xlabel('Target Classes')
plt.ylabel('Linear Discriminant 1')
plt.savefig('Hearth_Disease_Dataset.png')
plt.show()
