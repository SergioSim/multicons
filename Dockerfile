# -- Base image --
FROM python:3.9-slim-bullseye as base

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

RUN jupyter nbextension install --py jupytext && \
    jupyter nbextension enable --py jupytext

USER ${DOCKER_USER:-1000}
