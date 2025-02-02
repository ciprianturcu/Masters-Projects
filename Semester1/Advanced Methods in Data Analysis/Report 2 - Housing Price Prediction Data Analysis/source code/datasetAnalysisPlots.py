import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

housing_data = pd.read_csv('Housing.csv')  # Replace with the actual path to your CSV file

plt.style.use('ggplot')

numeric_features = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']

fig, axes = plt.subplots(len(numeric_features) // 2 + len(numeric_features) % 2, 2, figsize=(15, 15))
axes = axes.flatten()

for i, feature in enumerate(numeric_features):
    sns.histplot(housing_data[feature], kde=True, ax=axes[i], color="blue")
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('distributionOfNumericFeatures.png')
plt.show()

categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                        'airconditioning', 'prefarea', 'furnishingstatus']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    sns.countplot(data=housing_data, x=feature, palette="viridis", ax=axes[i])
    axes[i].set_title(f'Counts of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Count')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('countsOfCategoricalFeatures.png')
plt.show()

numeric_features = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
correlation_matrix = housing_data[numeric_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.savefig('correlationHeatmap.png')
plt.show()
