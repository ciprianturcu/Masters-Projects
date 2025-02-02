import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '../dataset/gym_members_exercise_tracking.csv'
df = pd.read_csv(file_path)

def plot_distributions(df):
    for column in df.columns:
        plt.figure(figsize=(10, 6))

        if pd.api.types.is_numeric_dtype(df[column]):  # For numeric columns
            sns.histplot(df[column], kde=True, bins=30, color='blue')
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
        elif isinstance(df[column].dtype, pd.CategoricalDtype) or df[
            column].dtype == 'object':  # For categorical columns
            df[column].value_counts().plot(kind='bar', color='orange')
            plt.title(f'Bar Plot of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
        else:  # For binary or boolean columns
            sns.countplot(x=df[column], color='green')
            plt.title(f'Count Plot of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')

        plt.tight_layout()
        safe_column_name = column.replace(' ', '_')
        plt.savefig(f'../output/distributions/{safe_column_name}_distribution.png', dpi=300)
        plt.show()

plot_distributions(df)
