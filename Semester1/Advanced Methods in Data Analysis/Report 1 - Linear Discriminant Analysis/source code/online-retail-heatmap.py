import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

dataset = pd.read_excel('../Online Retail.xlsx')

dataset = dataset.dropna(subset=['CustomerID'])
dataset['TotalPrice'] = dataset['Quantity'] * dataset['UnitPrice']
dataset['Country'] = LabelEncoder().fit_transform(dataset['Country'])

customer_data = dataset.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum',
    'Country': 'first'
}).reset_index()

X = customer_data[['Quantity', 'TotalPrice', 'Country']]
X_scaled = StandardScaler().fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

plt.figure(figsize=(10, 8))
sns.heatmap(X_scaled_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap: Online Retail Dataset')
plt.tight_layout()
plt.savefig('output/Online_Retail_Heatmap.png')
plt.show()
