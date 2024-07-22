install:
	python -m pip install --upgrade pip
	python -m pip install -e .
	python -m pip install -e ".[dev]"

install-no-cuda:
	python -m pip install --upgrade pip
	python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
	python -m pip install -e ".[dev]"
fmt:
	ruff format .
	ruff check .  --select I --fix

lint:
	ruff check .
	ruff format --check .
	mypy .

test:
	pytest -svx . --tb=native