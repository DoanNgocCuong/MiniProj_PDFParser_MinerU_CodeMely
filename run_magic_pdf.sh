#!/bin/bash

docker run --gpus all \
  -e PDF_PATH="$1" \
  -e OUTPUT_DIR="$2" \
  -e METHOD="$3" \
  -v "$(pwd)":/app \
  your-image-name