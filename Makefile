.PHONY: setup data train explain app

install:
	 pip install -r requirements.txt
	 pip install -e .

data:
	python scripts/download_dataset.py

train:
	python scripts/train_model.py

explain:
	python scripts/run_explainability.py

app:
	streamlit run student_performance_xai/app/app.py
