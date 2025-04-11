FROM python:3.11-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir setuptools==78.1.0 wheel==0.46.1 && python -m pip install -r requirements.txt

COPY . /app/

CMD ["./docker-entrypoint.sh"]