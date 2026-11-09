.PHONY: deps-torch deps deps-dev install lint format-check test migrate ingest manifest eval cost up down

deps-torch:
	python -m pip install -r requirements-torch.txt

deps:
	python -m pip install -r requirements.txt

deps-dev:
	python -m pip install -r requirements-dev.txt

install:
	python -m pip install -e . --no-deps

lint:
	python -m ruff check src tests alembic

format-check:
	python -m ruff format --check src tests alembic

test:
	python -m pytest

migrate:
	python -m alembic upgrade head

ingest:
	python -m supportdesk.ingest

manifest:
	python -m supportdesk.manifest

eval:
	python -m supportdesk.eval

cost:
	python -m supportdesk.cost

up:
	docker compose up -d --build

down:
	docker compose down
