set shell := ["bash", "-c"]

default: run

run:
	echo "Write this at some point"

dash:
	pipenv run python src/graddnodi/app.py
