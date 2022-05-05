# -- Docker
# Get the current user ID to use for docker run and docker exec commands
DOCKER_UID           = $(shell id -u)
DOCKER_GID           = $(shell id -g)
DOCKER_USER          = $(DOCKER_UID):$(DOCKER_GID)
COMPOSE              = DOCKER_USER=$(DOCKER_USER) docker-compose
COMPOSE_RUN          = $(COMPOSE) run --rm
COMPOSE_RUN_APP      = $(COMPOSE_RUN) app
MKDOCS               = $(COMPOSE_RUN) -e HOME=/app/.jupyterlab --publish "8000:8000" app mkdocs
JUPYTER              = $(COMPOSE_RUN) -e HOME=/app/.jupyterlab --publish "8888:8888"
COMPOSE_RUN_JUPYTER  = $(JUPYTER) app jupyter lab

default: help

# ======================================================================================
# FILES
# ======================================================================================

.jupyterlab:
	mkdir -p .jupyterlab

# ======================================================================================
# RULES
# ======================================================================================

build: ## build the docker container
	@$(COMPOSE) build
.PHONY: build

down: ## stop and remove the docker container
	rm -rf .jupyterlab
	@$(COMPOSE) down --rmi all -v --remove-orphans
.PHONY: down

docs-build: ## build documentation site
	@$(MKDOCS) build
.PHONY: docs-build

docs-deploy: ## deploy documentation site
	@$(MKDOCS) gh-deploy
.PHONY: docs-deploy

docs-serve: ## run mkdocs live server
	@$(MKDOCS) serve --dev-addr 0.0.0.0:8000
.PHONY: docs-serve

jupyter: \
	.jupyterlab
jupyter:  ## run jupyter lab
	@$(COMPOSE_RUN_JUPYTER) --notebook-dir=/app/docs --ip "0.0.0.0"
.PHONY: jupyter

test: ## run tests
	bin/pytest -vv
.PHONY: test

# Nota bene: Black should come after isort just in case they don't agree...
lint: ## lint back-end python sources
lint: \
  lint-isort \
  lint-black \
  lint-flake8 \
  lint-pylint \
  lint-bandit
.PHONY: lint

lint-black: ## lint back-end python sources with black
	@echo 'lint:black started…'
	@$(COMPOSE_RUN_APP) black src/multicons tests
.PHONY: lint-black

lint-flake8: ## lint back-end python sources with flake8
	@echo 'lint:flake8 started…'
	@$(COMPOSE_RUN_APP) flake8
.PHONY: lint-flake8

lint-isort: ## automatically re-arrange python imports in back-end code base
	@echo 'lint:isort started…'
	@$(COMPOSE_RUN_APP) isort --atomic .
.PHONY: lint-isort

lint-pylint: ## lint back-end python sources with pylint
	@echo 'lint:pylint started…'
	@$(COMPOSE_RUN_APP) pylint src/multicons tests
.PHONY: lint-pylint

lint-bandit: ## lint back-end python sources with bandit
	@echo 'lint:bandit started…'
	@$(COMPOSE_RUN_APP) bandit -qr src/multicons
.PHONY: lint-bandit

# -- Misc
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help
