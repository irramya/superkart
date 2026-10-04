# Lightweight Python base image
FROM python:3.11-slim

# Application directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Flask application and serialized model
COPY . .

# Flask/Gunicorn service port
EXPOSE 7860

# Start the API using Gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "app:superkart_api"]
