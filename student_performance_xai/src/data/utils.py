import pandas as pd

from student_performance_xai.config import RAW_DATA


def load_data(file_name="student-por.csv"):
    file_path = f"{RAW_DATA}/{file_name}"
    return pd.read_csv(file_path)
