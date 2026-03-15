FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required by LightGBM and scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (layer caching — only reinstalls if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the project package
RUN pip install --no-cache-dir -e .

# Create monitoring directory
RUN mkdir -p artifacts/monitoring

EXPOSE 8080

CMD ["python", "application.py"]
