SHELL = /bin/bash

.PHONY: setup
.PHONY: run
.PHONY: freeze
.PHONY: docs


setup:
	pipenv install --dev

run:
	pipenv run graddnodi

freeze:
	pipenv requirements --dev > requirements.txt

docs:
	pipenv run pdoc calidhayte -d numpy -o docs/ --math --mermaid --search
