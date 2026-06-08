FROM python:3.11-slim

WORKDIR /app

RUN echo "deb http://deb.debian.org/debian bookworm main contrib non-free non-free-firmware" \
        > /etc/apt/sources.list.d/non-free.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    intel-opencl-icd \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY prompt.txt .

CMD ["python", "src/main.py"]