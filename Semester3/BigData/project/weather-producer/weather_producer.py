import os
import time
import json
import logging
import requests
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
CITY = os.getenv('CITY', 'Cluj-Napoca')
COUNTRY = os.getenv('COUNTRY', 'RO')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', 300))
KAFKA_TOPIC = 'weather-data'

OPENWEATHER_URL = 'https://api.openweathermap.org/data/2.5/weather'


def create_kafka_producer():
    max_retries = 10
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            logger.info(f"Successfully connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
            return producer
        except Exception as e:
            logger.warning(f"Failed to connect to Kafka (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise


def fetch_weather_data():
    if not OPENWEATHER_API_KEY:
        raise ValueError("OPENWEATHER_API_KEY environment variable is not set!")

    params = {
        'q': f"{CITY},{COUNTRY}",
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric'
    }

    try:
        response = requests.get(OPENWEATHER_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        weather_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'temperature': data.get('main', {}).get('temp'),
            'humidity': data.get('main', {}).get('humidity'),
            'pressure': data.get('main', {}).get('pressure'),
        }

        return weather_record

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing weather data: {e}")
        return None


def send_to_kafka(producer, weather_data):
    try:
        key = "cluj-napoca"

        future = producer.send(KAFKA_TOPIC, key=key, value=weather_data)

        record_metadata = future.get(timeout=10)

        logger.info(
            f"Sent weather data to Kafka - "
            f"Topic: {record_metadata.topic}, "
            f"Partition: {record_metadata.partition}, "
            f"Offset: {record_metadata.offset}, "
            f"Temp: {weather_data['temperature']}°C, "
            f"Humidity: {weather_data['humidity']}%, "
            f"Pressure: {weather_data['pressure']}hPa"
        )
        return True

    except KafkaError as e:
        logger.error(f"Failed to send to Kafka: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending to Kafka: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("Weather Data Producer Starting")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  - City: {CITY}, {COUNTRY}")
    logger.info(f"  - Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"  - Topic: {KAFKA_TOPIC}")
    logger.info(f"  - Poll Interval: {POLL_INTERVAL} seconds")
    logger.info(f"  - API Key: {'✓ Set' if OPENWEATHER_API_KEY else '✗ Not Set'}")
    logger.info("=" * 60)

    producer = create_kafka_producer()

    try:
        while True:
            logger.info(f"Fetching weather data for {CITY}, {COUNTRY}...")

            weather_data = fetch_weather_data()

            if weather_data:
                success = send_to_kafka(producer, weather_data)
                if success:
                    logger.info(f"✓ Successfully processed weather update")
                else:
                    logger.warning(f"✗ Failed to send to Kafka")
            else:
                logger.warning(f"✗ Failed to fetch weather data")

            logger.info(f"Waiting {POLL_INTERVAL} seconds until next poll...")
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}", exc_info=True)
    finally:
        producer.close()
        logger.info("Producer closed. Exiting.")


if __name__ == '__main__':
    main()
