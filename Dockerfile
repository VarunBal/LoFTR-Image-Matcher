# Use a lightweight base image
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates templates

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]