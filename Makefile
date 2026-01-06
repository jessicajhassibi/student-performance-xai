.PHONY: setup data train explain app

install:
	 pip install -r requirements.txt
	 pip install -e .

data:
	python scripts/download_dataset.py

app:
	streamlit run student_performance_xai/app/app.py
