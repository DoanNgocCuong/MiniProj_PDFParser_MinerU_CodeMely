FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

WORKDIR /app

COPY requirements_fullCodeMely.txt .

# Lọc ra các package không tương thích với Linux
RUN grep -v pywin32 requirements_fullCodeMely.txt > requirements_linux.txt

# Cài đặt các package đã được lọc
RUN pip3 install --no-cache-dir -r requirements_linux.txt

RUN pip3 install detectron2 --extra-index-url https://myhloli.github.io/wheels/
RUN pip3 install magic-pdf[full] --extra-index-url https://myhloli.github.io/wheels/

# Copy mã nguồn custom PaddleOCR
COPY custom_paddleocr /app/custom_paddleocr

# Cài đặt custom PaddleOCR
RUN pip3 install -e /app/custom_paddleocr

COPY . .

ENV CUDA_VISIBLE_DEVICES=0
ENV PDF_PATH="/app/PDFTesting/testpdf2.pdf"
ENV OUTPUT_DIR="/app/output_directory"
ENV METHOD="ocr"

ENTRYPOINT ["magic-pdf", "--path", "$PDF_PATH", "--output-dir", "$OUTPUT_DIR", "--method", "$METHOD"]