import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


selected_columns = ["sex"]

def univariate_analysis(df: pd.DataFrame, cols=None):
	figs = []
	if cols is None:
		cols = df.select_dtypes(include=['object']).columns
	for col in cols:
		fig, ax = plt.subplots()
		sns.countplot(data=df, x=col)
		ax.set_title(f"Distribution of {col}")
		figs.append(fig)
	return figs

def bivariate_analysis(df: pd.DataFrame, col1: str, col2: str) -> tuple:
	fig, ax = plt.subplots()
	sns.boxplot(x=col1, y=col2, data=df, ax=ax)
	ax.set_title(f"{col2} by {col1}")
	return fig, ax