# Vector Database Demonstration

This repository is a demonstration of vector database using OpenSearch and Python

## Prerequisites

- [Docker](https://docs.docker.com/engine/install/)
- [OpenSearch](https://opensearch.org/docs/2.15/install-and-configure/install-opensearch/docker)
- [Python](https://www.python.org/downloads/)

## Install Dependencies

```
pip install -r requirements.txt
```

## How to Run

Let's look at Makefile to run a command that we need.

### Run OpenSearch

Execute `make upOS` to turn on our OpenSearch locally.

### Running API Server

Execute `make run` to run our API Server.

### Stop OpenSearch

Execute `make downOS` to teardown running OpenSearch.
