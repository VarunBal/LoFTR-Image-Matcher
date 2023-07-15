# Use a lightweight base image
FROM python:3.10-slim-buster

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import kornia;  \
    kornia.feature.LoFTR(pretrained='outdoor');  \
    kornia.feature.LoFTR(pretrained='indoor');  \
    kornia.feature.LoFTR(pretrained='indoor_new')"

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]