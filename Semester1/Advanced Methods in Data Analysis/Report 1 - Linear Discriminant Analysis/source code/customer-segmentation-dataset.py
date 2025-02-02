import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

dataset = pd.read_excel('Online Retail.xlsx')

dataset = dataset.dropna(subset=['CustomerID'])

le = LabelEncoder()
dataset['Country'] = le.fit_transform(dataset['Country'])

dataset['TotalPrice'] = dataset['Quantity'] * dataset['UnitPrice']

customer_data = dataset.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum',
    'Country': 'first'
}).reset_index()

X = customer_data[['Quantity', 'TotalPrice', 'Country']]

X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

y = customer_data['Cluster']

n_features = X_scaled.shape[1]
n_classes = len(np.unique(y))

n_components = min(n_features, n_classes - 1)
lda = LDA(n_components=n_components)
X_r2 = lda.fit_transform(X_scaled, y)

plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange', 'red', 'green']
target_names = [f'Cluster {i}' for i in range(5)]

for color, i, target_name in zip(colors, np.unique(y), target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name)

plt.title('LDA of Online Retail Dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='best')
plt.grid()
plt.savefig("Online_Retail_Dataset.png")
plt.show()