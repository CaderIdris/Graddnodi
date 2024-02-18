set shell := ["bash", "-c"]

default: run

run:
	echo "Write this at some point"

summary:
	pipenv run python src/graddnodi/summary_dashboard.py

pipeline:
	pipenv run python src/graddnodi/pipeline_dashboard.py
