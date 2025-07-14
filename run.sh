#!/bin/bash

echo "Starting FastAPI app..."

if [[ "$1" == "--debug" ]]; then
  echo "Running in DEBUG mode"
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
else
  echo "Running in PRODUCTION mode"
  uvicorn main:app --host 0.0.0.0 --port 8000
fi
