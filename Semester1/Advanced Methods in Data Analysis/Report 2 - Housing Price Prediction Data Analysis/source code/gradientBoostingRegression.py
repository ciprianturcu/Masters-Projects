import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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

gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_regressor.fit(X_train, y_train)

y_pred = gb_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Gradient Boosting Regression MSE: {mse}')
print(f'Gradient Boosting Regression R^2 Score: {r2}')

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Gradient Boosting Regression: Actual vs Predicted Values')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')

plt.savefig('gradientBoostingRegressionPlot.png', dpi=300, bbox_inches='tight')
plt.show()
