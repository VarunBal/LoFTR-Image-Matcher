# Use a lightweight base image
FROM python:3.10-slim-buster

WORKDIR /app

COPY . .

RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Download Kornia LoFTR models
RUN python -c "import kornia;  \
    kornia.feature.LoFTR(pretrained='outdoor');  \
    kornia.feature.LoFTR(pretrained='indoor');  \
    kornia.feature.LoFTR(pretrained='indoor_new')"

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["flask", "run", "--host=0.0.0.0"]