import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

heart_data = pd.read_csv('../HeartDiseaseTrain-Test.csv')

categorical_cols = heart_data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    heart_data[col] = le.fit_transform(heart_data[col])

X_heart = heart_data.drop('target', axis=1)
X_scaled = StandardScaler().fit_transform(X_heart)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_heart.columns)

plt.figure(figsize=(12, 10))
sns.heatmap(X_scaled_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap: Heart Disease Dataset')
plt.tight_layout()
plt.savefig('output/Heart_Disease_Heatmap.png')
plt.show()
