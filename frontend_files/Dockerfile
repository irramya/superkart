# Lightweight Python base image
FROM python:3.11-slim

# Application working directory
WORKDIR /app

# Install frontend dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Streamlit application
COPY . .

# Standard Streamlit port
EXPOSE 8501

# Start Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
