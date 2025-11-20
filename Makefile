.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")

.PHONY: help
help:             	## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: venv
venv:			## Create a virtual environment
	@echo "Creating virtualenv ..."
	@rm -rf .venv
	@python3 -m venv .venv
	@./.venv/bin/pip install -U pip
	@echo
	@echo "Run 'source .venv/bin/activate' to enable the environment"

.PHONY: install
install:		## Install dependencies
	pip install -r requirements-dev.txt
	pip install -r requirements-test.txt
	pip install -r requirements.txt

STRESS_URL = https://delay-api-266724359764.us-central1.run.app/ 
.PHONY: stress-test
stress-test:
	# change stress url to your deployed app 
	mkdir reports || true
	locust -f tests/stress/api_stress.py --print-stats --html reports/stress-test.html --run-time 60s --headless --users 100 --spawn-rate 1 -H $(STRESS_URL)

.PHONY: model-test
model-test:			## Run tests and coverage
	mkdir reports || true
	pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/model

.PHONY: api-test
api-test:			## Run tests and coverage
	mkdir reports || true
	pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/api

.PHONY: build
build:			## Build locally the python artifact
	python setup.py bdist_wheel

# Docker targets
IMAGE_NAME = delay-api
IMAGE_TAG = latest

.PHONY: docker-build
docker-build:		## Build Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

.PHONY: docker-run
docker-run:		## Run Docker container locally
	docker run -p 8000:8000 --name $(IMAGE_NAME) $(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: docker-stop
docker-stop:		## Stop and remove Docker container
	docker stop $(IMAGE_NAME) || true
	docker rm $(IMAGE_NAME) || true

.PHONY: docker-test
docker-test:		## Test Docker image locally
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME):test .
	@echo "Starting container..."
	docker run -d --name test-container -p 8000:8000 $(IMAGE_NAME):test
	@echo "Waiting for service to start..."
	sleep 10
	@echo "Testing health endpoint..."
	curl -f http://localhost:8000/health || (docker stop test-container && docker rm test-container && exit 1)
	@echo "Testing predict endpoint..."
	curl -f -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"flights":[{"OPERA":"Aerolineas Argentinas","TIPOVUELO":"N","MES":3}]}' \
		|| (docker stop test-container && docker rm test-container && exit 1)
	@echo "Cleaning up..."
	docker stop test-container
	docker rm test-container
	@echo "✅ Docker tests passed!"

.PHONY: docker-clean
docker-clean:		## Remove Docker images and containers
	docker stop $(IMAGE_NAME) || true
	docker rm $(IMAGE_NAME) || true
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) || true
	docker rmi $(IMAGE_NAME):test || true