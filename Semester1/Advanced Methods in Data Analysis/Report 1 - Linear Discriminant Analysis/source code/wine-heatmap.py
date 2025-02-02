import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

wine_data = pd.read_csv('../WineQT.csv')

X_wine = wine_data.drop(columns=['quality'])
X_scaled = StandardScaler().fit_transform(X_wine)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_wine.columns)

plt.figure(figsize=(10, 8))
sns.heatmap(X_scaled_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap: Wine Quality Dataset')
plt.tight_layout()
plt.savefig('output/Wine_Quality_Heatmap.png')
plt.show()
