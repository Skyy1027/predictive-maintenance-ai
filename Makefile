.PHONY: install train validate run docker-build

install:
	python3 -m pip install -r requirements.txt

train:
	python3 train_model.py

validate:
	python3 validate_project.py

run:
	python3 -m streamlit run app.py

docker-build:
	docker build -t maintainai:latest .
