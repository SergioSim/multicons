# -- Base image --
FROM python:3.10-slim-bullseye as base

RUN pip install --upgrade pip

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y gcc git graphviz python3-dev && \
    rm -rf /var/lib/apt/lists/*

# -- Development --
FROM base as development

WORKDIR /app

COPY . /app/

RUN pip install -e .[dev]

RUN jupyter contrib nbextension install && \
   jupyter nbextension install jupytext --py && \
   jupyter nbextension enable jupytext --py

USER ${DOCKER_USER:-1000}
