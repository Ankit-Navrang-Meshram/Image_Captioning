FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (git for CLIP)
RUN apt-get update && apt-get install -y git

# Install Python libs
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers pillow fastapi uvicorn python-multipart
RUN pip install git+https://github.com/openai/CLIP.git

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]