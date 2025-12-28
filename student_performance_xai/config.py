# Here lie the configuration settings and often used variables for the project.
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA = os.path.join(BASE_DIR, "student_performance_xai", "data", "raw")
PROCESSED_DATA = os.path.join(BASE_DIR, "student_performance_xai", "data", "processed")
KAGGLE_LOCATION = os.path.join(BASE_DIR, "venv", "bin", "kaggle")
KAGGLE_DATA = "larsen0966/student-performance-data-set"
