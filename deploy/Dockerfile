FROM ubuntu:20.04

RUN apt-get update && apt-get install pip -y
COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
