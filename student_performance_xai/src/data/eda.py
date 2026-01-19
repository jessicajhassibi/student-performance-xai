import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


selected_columns = ["sex"]

def univariate_analysis(df: pd.DataFrame, selected_columns=None):
	figs = []
	if selected_columns is None:
		selected_columns = df.select_dtypes(include=['object']).columns
	for col in selected_columns:
		fig, ax = plt.subplots()
		sns.countplot(data=df, x=col)
		ax.set_title(f"Distribution of {col}")
		figs.append(fig)
		plt.close(fig)
	return figs