import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

file_path = '../dataset/gym_members_exercise_tracking.csv'
df = pd.read_csv(file_path)

for col in ['Gender', 'Workout_Type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

numerical_columns = df.columns.tolist()

columns_part1 = numerical_columns[:8]
columns_part2 = numerical_columns[8:]

pairplot_part1 = sns.pairplot(
    df,
    x_vars=columns_part1,
    y_vars=numerical_columns,
    diag_kind='hist',
    plot_kws={'alpha': 0.5, 's': 10}
)
pairplot_part1.fig.set_size_inches(20, 20)
pairplot_part1.fig.subplots_adjust(wspace=0.3, hspace=0.3)
pairplot_part1.savefig("../output/pairplots/pairplot_part1.png", dpi=100)
plt.show()


pairplot_part2 = sns.pairplot(
    df,
    x_vars=columns_part2,
    y_vars=numerical_columns,
    diag_kind='hist',
    plot_kws={'alpha': 0.5, 's': 10}
)
pairplot_part2.fig.set_size_inches(20, 20)
pairplot_part2.fig.subplots_adjust(wspace=0.3, hspace=0.3)
pairplot_part2.savefig("../output/pairplots/pairplot_part2.png", dpi=100)
plt.show()
