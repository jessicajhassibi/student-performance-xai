import pandas as pd

CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup",
	"paid", "activities", "nursery", "higher", "internet", "romantic"
]


def preprocess(df: pd.DataFrame):
    X_raw = df.drop("G3", axis=1)
    y = df["G3"]

    X = pd.get_dummies(X_raw, columns=CATEGORICAL_FEATURES, drop_first=True)
    return X, y