FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && python -m fraudml.train --n 12000
ENV MODEL_PATH=/app/artifacts/model.joblib
EXPOSE 8000
USER 65532
CMD ["uvicorn", "fraudml.api:app", "--host", "0.0.0.0", "--port", "8000"]
