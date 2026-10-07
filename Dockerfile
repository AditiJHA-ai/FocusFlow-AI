FROM python:3.11-slim

# Native libraries needed by OpenCV / MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    PORT=7860
WORKDIR /app

# CPU-only PyTorch keeps the image small (no CUDA libraries)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # ultralytics pulls in the GUI build of OpenCV; keep only the headless one
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps "opencv-python-headless>=4.9"

# YOLO weights are baked into the image so startup needs no download
RUN curl -fsSL -o /app/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

COPY --chown=user . /app
RUN chown -R user /app
USER user

EXPOSE 7860
# Flask-SocketIO's threading mode needs a single worker; threads serve concurrent visitors
CMD gunicorn --worker-class gthread --workers 1 --threads 32 --timeout 0 --bind 0.0.0.0:${PORT} app:app
