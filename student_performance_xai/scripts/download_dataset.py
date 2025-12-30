import os
import subprocess

from student_performance_xai.config import RAW_DATA, KAGGLE_LOCATION, KAGGLE_DATA


def download_dataset():
    os.makedirs(RAW_DATA, exist_ok=True)

    command = [
        KAGGLE_LOCATION,
        "datasets",
        "download",
        "-d", KAGGLE_DATA,
        "-p", RAW_DATA,
        "--unzip"
    ]

    subprocess.run(command, check=True)
    print("Dataset successfully downloaded!")
