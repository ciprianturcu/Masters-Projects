FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    kafka-python==2.0.2 \
    requests==2.31.0

# Copy application
COPY weather_producer.py .

# Run the producer
CMD ["python", "-u", "weather_producer.py"]