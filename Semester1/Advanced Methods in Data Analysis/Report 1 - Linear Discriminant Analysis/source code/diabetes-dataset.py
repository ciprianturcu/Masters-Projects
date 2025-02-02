import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

diabetes_data = pd.read_csv('diabetes.csv')

X_diabetes = diabetes_data.drop('Outcome', axis=1)
y_diabetes = diabetes_data['Outcome']

n_features = X_diabetes.shape[1]
n_classes = len(np.unique(y_diabetes))

n_components = min(n_features, n_classes - 1)

X_diabetes = StandardScaler().fit_transform(X_diabetes)

lda = LDA(n_components=n_components)
X_r2 = lda.fit(X_diabetes, y_diabetes).transform(X_diabetes)

lda_df = pd.DataFrame({'LD1': X_r2[:, 0], 'Target': y_diabetes})

plt.figure(figsize=(10, 6))
sns.stripplot(x='Target', y='LD1', data=lda_df, jitter=True, palette='Set1', alpha=0.7)
plt.title('LDA of Diabetes Dataset')
plt.xlabel('Target Classes')
plt.ylabel('Linear Discriminant 1')
plt.savefig('Diabetes_Dataset.png')
plt.show()
