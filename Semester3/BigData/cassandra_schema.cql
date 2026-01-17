-- Simplified Cassandra Schema for Weather Real-Time Data
-- Run this inside Cassandra container: docker exec -it cassandra cqlsh -f /path/to/this/file

-- Create keyspace
CREATE KEYSPACE IF NOT EXISTS weather 
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE weather;

-- Table for real-time weather data (only essential fields)
CREATE TABLE IF NOT EXISTS weather_realtime (
    timestamp timestamp PRIMARY KEY,
    temperature double,
    humidity double,
    pressure double
);

-- Table for rain predictions (weather data + prediction)
CREATE TABLE IF NOT EXISTS rain_predictions (
    timestamp timestamp PRIMARY KEY,
    temperature double,
    humidity double,
    pressure double,
    predicted_rain double
);

-- Create keyspace
CREATE KEYSPACE IF NOT EXISTS weather 
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE weather;

-- 1. Monthly Trends Table
CREATE TABLE IF NOT EXISTS monthly_trends (
    year INT,
    month INT,
    avg_temperature DOUBLE,
    min_temperature DOUBLE,
    max_temperature DOUBLE,
    stddev_temperature DOUBLE,
    total_rain_mm DOUBLE,
    avg_rain_per_hour DOUBLE,
    max_rain_per_hour DOUBLE,
    total_precipitation_mm DOUBLE,
    rainy_hours BIGINT,
    total_hours BIGINT,
    rain_percentage DOUBLE,
    PRIMARY KEY (year, month)
);

-- 2. Seasonal Trends Table
CREATE TABLE IF NOT EXISTS seasonal_trends (
    year INT,
    season TEXT,
    avg_temperature DOUBLE,
    min_temperature DOUBLE,
    max_temperature DOUBLE,
    total_rain_mm DOUBLE,
    rainy_hours BIGINT,
    total_hours BIGINT,
    rain_percentage DOUBLE,
    PRIMARY KEY (year, season)
);

-- 3. Hourly Patterns Table
CREATE TABLE IF NOT EXISTS hourly_patterns (
    hour_of_day INT PRIMARY KEY,
    avg_temperature DOUBLE,
    avg_rain DOUBLE,
    avg_precipitation DOUBLE,
    rainy_hours BIGINT,
    total_hours BIGINT,
    rain_probability DOUBLE
);

-- 4. Day of Week Patterns Table
CREATE TABLE IF NOT EXISTS dow_patterns (
    day_of_week INT PRIMARY KEY,
    day_name TEXT,
    avg_temperature DOUBLE,
    avg_rain DOUBLE,
    rainy_hours BIGINT,
    total_hours BIGINT,
    rain_probability DOUBLE
);

-- 5. Temperature Extremes Table
CREATE TABLE IF NOT EXISTS temp_extremes (
    year INT,
    month INT,
    coldest_temp DOUBLE,
    hottest_temp DOUBLE,
    temp_range DOUBLE,
    PRIMARY KEY (year, month)
);

-- 6. Overall Summary Table
CREATE TABLE IF NOT EXISTS overall_summary (
    id BIGINT PRIMARY KEY,
    avg_temperature DOUBLE,
    min_temperature DOUBLE,
    max_temperature DOUBLE,
    stddev_temperature DOUBLE,
    total_rain_mm DOUBLE,
    avg_rain_per_hour DOUBLE,
    max_rain_hour DOUBLE,
    total_rainy_hours BIGINT,
    total_hours BIGINT,
    rain_percentage DOUBLE
);

-- Display tables
DESCRIBE TABLES;

SELECT * FROM system_schema.tables WHERE keyspace_name = 'weather';


