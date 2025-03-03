.PHONY: setup test clean lint run

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -r {} +

lint:
	flake8 src/ tests/
	black src/ tests/ --check

run:
	python src/pipeline.py