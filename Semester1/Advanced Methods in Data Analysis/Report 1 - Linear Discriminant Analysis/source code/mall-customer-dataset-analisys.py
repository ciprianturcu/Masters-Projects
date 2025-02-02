import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

dataset = pd.read_csv('Mall_Customers.csv')

le = LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset['Gender'])

X = dataset[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
dataset['Cluster'] = kmeans.fit_predict(X_scaled)

y = dataset['Cluster']

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

plt.title('LDA of Mall Customer Dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='best')
plt.grid()
plt.savefig('Mall_Customers_Dataset.png')
plt.show()
