FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m ipykernel install --user --name python3

COPY config.py llm.py pipeline.py prompts.py notebook_builder.py \
     generate_notebook.py web_pipeline.py app.py ./
COPY static/ static/

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
