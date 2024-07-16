install:
	python -m pip install --upgrade pip
	python -m pip install -e .
	python -m pip install -e ".[dev]"

fmt:
	ruff format .
	ruff check .  --select I --fix

lint:
	ruff check .
	ruff format --check .
	mypy .

test:
	pytest -vx . --tb=native