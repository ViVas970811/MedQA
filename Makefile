.PHONY: install dev lint test api frontend docker-up docker-down evaluate clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/ frontend/
	ruff format --check src/ tests/ frontend/

format:
	ruff format src/ tests/ frontend/

test:
	pytest tests/ -v --tb=short

api:
	uvicorn medqa.api.app:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/app.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

evaluate:
	python scripts/evaluate.py

build-index:
	python scripts/build_index.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
