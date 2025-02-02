import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

dataset = pd.read_csv('../Mall_Customers.csv')

dataset['Gender'] = LabelEncoder().fit_transform(dataset['Gender'])
X = dataset[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
X_scaled = StandardScaler().fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

plt.figure(figsize=(10, 8))
sns.heatmap(X_scaled_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap: Mall Customer Dataset')
plt.tight_layout()
plt.savefig('output/Mall_Customer_Heatmap.png')
plt.show()
