import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

diabetes_data = pd.read_csv('../diabetes.csv')

X_diabetes = diabetes_data.drop('Outcome', axis=1)
X_scaled = StandardScaler().fit_transform(X_diabetes)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_diabetes.columns)

plt.figure(figsize=(10, 8))
sns.heatmap(X_scaled_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap: Diabetes Dataset')
plt.tight_layout()
plt.savefig('output/frdioppppppppppppppppppppppppppppppppppppppppppppppppppppppppppllllllllllllllllllllllllllllllllllDiabetes_Heatmap.png')
plt.show()
