# syntax=docker/dockerfile:1

FROM python:3.9

RUN apt-get update && apt-get -y install sudo
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY aic /Project_AIC/aic
COPY scripts /Project_AIC/scripts

COPY setup.py requirements-docker.txt README.md /Project_AIC/
WORKDIR /Project_AIC
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install -r requirements-docker.txt
RUN python3.9 -m pip install -e . --user

WORKDIR /Project_AIC/scripts/web