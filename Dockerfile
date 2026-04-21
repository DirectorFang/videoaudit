FROM python:3.11
#FROM hub-mirror.c.163.com/library/python:3.11

WORKDIR /app

ENV POETRY_VERSION=1.8.3

RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

RUN poetry config virtualenvs.create false

#COPY pyproject.toml poetry.lock* /app/
COPY pyproject.toml /app/pyproject.toml

RUN echo "===== FILE PATH CHECK =====" && ls -lah /app
RUN echo "===== FILE CONTENT =====" && cat /app/pyproject.toml
RUN echo "===== FILE TYPE CHECK =====" && file /app/pyproject.toml
RUN cat pyproject.toml

RUN poetry install --no-interaction --no-ansi --no-root

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "APP:app", "--host", "0.0.0.0", "--port", "8000"]
