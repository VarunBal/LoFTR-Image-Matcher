# Use a lightweight base image
FROM python:3.10-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y wget \
    && wget https://download.pytorch.org/whl/cpu/torch-2.0.1%2Bcpu-cp310-cp310-linux_x86_64.whl \
    && pip install torch-2.0.1+cpu-cp310-cp310-linux_x86_64.whl \
    && rm torch-2.0.1+cpu-cp310-cp310-linux_x86_64.whl

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