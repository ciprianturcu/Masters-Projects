import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('housing.csv')

print(data.info())
print(data.describe())

print(data.isnull().sum())

#data = data.dropna()

categorical_cols = data.select_dtypes(include=['object']).columns
print(f'Categorical columns: {categorical_cols}')

data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train_scaled, y_train)

y_pred = knn_regressor.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'K-Nearest Neighbors Regression MSE: {mse}')
print(f'K-Nearest Neighbors Regression R^2 Score: {r2}')

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('K-Nearest Neighbors Regression: Actual vs Predicted Values')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')

plt.savefig('kNearestNeighborsRegressionPlot.png', dpi=300, bbox_inches='tight')

plt.show()
