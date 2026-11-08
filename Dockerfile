FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-torch.txt requirements.txt pyproject.toml README.md alembic.ini ./
COPY src ./src
COPY alembic ./alembic

RUN python -m pip install --no-cache-dir -r requirements-torch.txt \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir -e . --no-deps

COPY data ./data

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "supportdesk.main:app", "--host", "0.0.0.0", "--port", "8000"]
