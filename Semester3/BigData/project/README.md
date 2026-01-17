# Weather Prediction Pipeline

A Big Data pipeline for weather data collection, historical trend analysis, and rain prediction using Apache Spark, Kafka, Cassandra, and XGBoost.

## Team Project

This was a collaborative team project. My contributions included:
- **Architecture Design** - Designed the service hosting structure and inter-service communication
- **Grafana Dashboards** - Created visualization dashboards for real-time and historical data
- **Weather Producer** - Built the Kafka producer for live weather data collection from OpenWeather API
- **Data Ingestion Scripts** - Developed the data ingestion pipeline for loading historical data into HDFS

## Architecture

```
OpenWeather API → Kafka → Spark Streaming → Cassandra → Grafana
                            ↓
Historical Data → HDFS → Spark ML (XGBoost) → Model
```

### Components

| Component | Purpose | Port |
|-----------|---------|------|
| HDFS (Namenode) | Distributed storage | 9870, 9000 |
| HDFS (Datanodes x3) | Data replication | 9864-9866 |
| Spark Master | Distributed processing | 8081, 7077 |
| Spark Workers (x2) | Job execution | - |
| Kafka | Message streaming | 9092, 29092 |
| Zookeeper | Kafka coordination | 2181 |
| Cassandra | Time-series database | 9042 |
| Grafana | Visualization | 3000 |
| Weather Producer | API data collection | - |

## Project Structure

```
BigData/
├── hadoop-compose.yml          # Docker Compose configuration
├── cassandra_schema.cql        # Cassandra table definitions
├── historicalExctraction.py    # Fetch historical data from Open-Meteo API
├── .env                        # Environment variables (API keys)
├── data/
│   └── historical_weather_data.csv
├── spark-apps/
│   ├── weather_data_ingestion.py      # Load CSV to HDFS
│   ├── model_training.py              # XGBoost rain prediction model
│   ├── historical_trend_analysis.py   # Compute trends for Grafana
│   └── streaming_consumer.py          # Real-time predictions
├── weather-producer/
│   ├── Dockerfile
│   └── weather_producer.py            # Kafka producer for live weather
├── grafana-dashboard-realtime.json    # Real-time dashboard config
├── grafana-dashboard-trend-analysis.json  # Historical trends dashboard
├── requirements.txt                   # Python dependencies
└── xgboost_model_from_hdfs/           # Exported trained model
```

## Quick Start

### 1. Start the Cluster

```bash
docker-compose -f hadoop-compose.yml up -d
```

### 2. Initialize Cassandra Schema

Wait ~30 seconds for Cassandra to start, then run:

**Windows (PowerShell):**
```powershell
Get-Content cassandra_schema.cql | docker exec -i cassandra cqlsh
```

**Linux/Mac:**
```bash
docker exec -i cassandra cqlsh < cassandra_schema.cql
```

### 3. Verify Tables

```bash
docker exec cassandra cqlsh -k weather -e "DESCRIBE TABLES"
```

**Expected tables:**
- `monthly_trends` - Monthly aggregations
- `seasonal_trends` - Seasonal patterns
- `hourly_patterns` - Hourly rain probability
- `dow_patterns` - Day-of-week patterns
- `temp_extremes` - Temperature extremes
- `overall_summary` - Overall statistics
- `rain_predictions` - Real-time predictions
- `weather_realtime` - Live weather data

### 4. Install Spark Dependencies

```bash
docker exec spark-master pip install numpy xgboost pandas scikit-learn pyarrow
docker exec spark-worker1 pip install numpy xgboost pandas scikit-learn pyarrow
docker exec spark-worker2 pip install numpy xgboost pandas scikit-learn pyarrow
```

### 5. Data Ingestion

Load historical weather data into HDFS:

```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --packages com.datastax.spark:spark-cassandra-connector_2.12:3.4.1 \
  /opt/spark-apps/weather_data_ingestion.py
```

### 6. Generate Historical Trends

Compute and save trends to Cassandra:

```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --packages com.datastax.spark:spark-cassandra-connector_2.12:3.4.1 \
  /opt/spark-apps/historical_trend_analysis.py
```

### 7. Train ML Model (Optional)

Train the XGBoost rain prediction model:

```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --packages com.datastax.spark:spark-cassandra-connector_2.12:3.4.1 \
  /opt/spark-apps/model_training.py
```

## Historical Data Extraction

To fetch fresh historical weather data from Open-Meteo API:

```bash
python -m venv venv
.\venv\Scripts\Activate      # Windows PowerShell
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
python historicalExctraction.py
```

This downloads hourly weather data (temperature, rain, humidity, pressure) and saves it to `historical_weather_data.csv`.

## Model Management

### Export Model from HDFS

```bash
docker exec namenode hdfs dfs -get /user/hadoop/weather_prediction/models/xgboost /tmp/xgboost_model
docker cp namenode:/tmp/xgboost_model ./xgboost_model_from_hdfs
```

### Import Model to HDFS

```bash
docker cp ./xgboost_model_from_hdfs namenode:/tmp/xgboost_model
docker exec namenode hdfs dfs -mkdir -p /user/hadoop/weather_prediction/models
docker exec namenode hdfs dfs -put /tmp/xgboost_model /user/hadoop/weather_prediction/models/xgboost
```

### Verify Model Location

```bash
docker exec namenode hdfs dfs -ls /user/hadoop/weather_prediction/models/xgboost
```

## Grafana Visualization

### Access Grafana

1. Open: `http://localhost:3000`
2. Login: `admin` / `admin`

### Configure Cassandra Data Source

1. Go to **Configuration** → **Data Sources** → **Add data source**
2. Search for **Apache Cassandra** (HadesArchitect plugin)
3. Configure:
   - **Host:** `cassandra:9042`
   - **Keyspace:** `weather`
4. Click **Save & Test**

### Import Dashboards

Import the pre-configured dashboards:
- `grafana-dashboard-realtime.json` - Real-time weather and predictions
- `grafana-dashboard-trend-analysis.json` - Historical trend analysis

### Example Queries

**Recent Rain Predictions:**
```sql
SELECT timestamp, temperature, predicted_rain
FROM rain_predictions
LIMIT 1000
ALLOW FILTERING
```

**Monthly Temperature Trends:**
```sql
SELECT year, month, avg_temperature, total_rain_mm
FROM monthly_trends
```

## Troubleshooting

### Check Container Status

```bash
docker ps
```

### Network Connectivity Issues

Verify Cassandra is on the correct network:

```bash
docker network ls
docker network inspect bigdata_weather-network
```

If Cassandra is not connected:

```bash
docker network connect bigdata_weather-network cassandra
```

### Container Management

```bash
docker-compose -f hadoop-compose.yml stop      
docker-compose -f hadoop-compose.yml start     
docker-compose -f hadoop-compose.yml down      
docker-compose -f hadoop-compose.yml up -d     
```

### Restart Individual Services

```bash
docker-compose -f hadoop-compose.yml restart cassandra
docker-compose -f hadoop-compose.yml restart spark-master
```

## Environment Variables

Create a `.env` file with:

```
OPENWEATHER_API_KEY=your_api_key_here
```

Get an API key from [OpenWeatherMap](https://openweathermap.org/api).

## Web Interfaces

| Service | URL |
|---------|-----|
| HDFS Namenode | http://localhost:9870 |
| Spark Master | http://localhost:8081 |
| Grafana | http://localhost:3000 |
