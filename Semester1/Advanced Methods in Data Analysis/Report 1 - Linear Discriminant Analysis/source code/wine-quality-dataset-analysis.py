import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

wine_data = pd.read_csv('WineQT.csv')

X_wine = wine_data.drop(columns=['quality'])
y_wine = wine_data['quality']

n_features = X_wine.shape[1]
n_classes = len(np.unique(y_wine))

n_components = min(n_features, n_classes - 1)

X_wine = StandardScaler().fit_transform(X_wine)

lda = LDA(n_components=n_components)
X_r2 = lda.fit(X_wine, y_wine).transform(X_wine)

plt.figure()
colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
target_names_wine = sorted(np.unique(y_wine))
for color, i, target_name in zip(colors, np.unique(y_wine), target_names_wine):
    plt.scatter(X_r2[y_wine == i, 0], X_r2[y_wine == i, 1], alpha=.8, color=color,
                label=f'Quality {target_name}')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of Wine Quality Dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.savefig('Wine_Quality_Dataset.png')
plt.show()
