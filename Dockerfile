FROM python:3.12-slim
WORKDIR /app
COPY . .
ENTRYPOINT ["python", "prompt_profiler.py"]
