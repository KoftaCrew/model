FROM ubuntu:20.04

RUN apt-get update && apt-get install pip -y
COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

ADD assets/ inference/assets
ADD model/ inference/model
ADD app.py inference/app.py

WORKDIR inference/
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
