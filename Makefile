.PHONY: install install-dev train app test lint format clean docker-build docker-run

# ── Setup ──────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# ── Application ────────────────────────────────────────────────
app:
	python -m app.main

# ── Training ───────────────────────────────────────────────────
train:
	python -m src.training.trainer

# ── Testing ────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=html

# ── Code Quality ───────────────────────────────────────────────
lint:
	flake8 src/ app/ tests/
	mypy src/ app/

format:
	black src/ app/ tests/
	isort src/ app/ tests/

# ── Cleanup ────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .mypy_cache

# ── Docker ─────────────────────────────────────────────────────
docker-build:
	cd docker && docker-compose build

docker-run:
	cd docker && docker-compose up

# ── Data ───────────────────────────────────────────────────────
download-data:
	python -m src.data.dataset
