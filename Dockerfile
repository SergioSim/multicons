# -- Base image --
FROM python:3.12-slim-bullseye as base

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

USER ${DOCKER_USER:-1000}
