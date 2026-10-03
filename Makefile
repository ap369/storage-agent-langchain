.PHONY: run

run:
	uv run uvicorn main:app --port 8000
