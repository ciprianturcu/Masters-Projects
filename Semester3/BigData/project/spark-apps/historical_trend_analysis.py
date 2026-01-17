from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, sum as spark_sum, min as spark_min, max as spark_max,
    count, stddev, when, hour, dayofweek, month, year,
    round as spark_round, percentile_approx, corr, lit
)
from pyspark.sql.window import Window
import time

spark = SparkSession.builder \
    .appName("Historical Weather Trend Analysis") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.cassandra.connection.port", "9042") \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.memory", "1g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 80)
print("HISTORICAL WEATHER TREND ANALYSIS")
print("=" * 80)
print("Reading from: HDFS")
print("Writing to: Cassandra (weather keyspace)")
print("Purpose: Generate trends for Grafana visualization")
print("=" * 80)

HDFS_INPUT_PATH = "hdfs://namenode:9000/user/hadoop/weather_prediction/data/historical_weather"

print("\nReading historical weather data from HDFS...")
df = spark.read.parquet(HDFS_INPUT_PATH)

total_records = df.count()
print(f"Total records: {total_records:,}")

date_stats = df.agg(
    spark_min("timestamp").alias("start_date"),
    spark_max("timestamp").alias("end_date")
).collect()[0]

print(f"Date range: {date_stats['start_date']} to {date_stats['end_date']}")
df.printSchema()

print("\n" + "=" * 80)
print("COMPUTING TRENDS")
print("=" * 80)

print("\n" + "=" * 80)
print("PROCESSING TRENDS (ONE AT A TIME TO SAVE MEMORY)")
print("=" * 80)
print("Total trends to compute: 6")

print("\n[1/6] Computing and saving monthly trends...")

monthly_trends = df.groupBy("year", "month").agg(
    spark_round(avg("temperature_2m"), 2).alias("avg_temperature"),
    spark_round(spark_min("temperature_2m"), 2).alias("min_temperature"),
    spark_round(spark_max("temperature_2m"), 2).alias("max_temperature"),
    spark_round(stddev("temperature_2m"), 2).alias("stddev_temperature"),
    spark_round(spark_sum("rain"), 2).alias("total_rain_mm"),
    spark_round(avg("rain"), 4).alias("avg_rain_per_hour"),
    spark_round(spark_max("rain"), 2).alias("max_rain_per_hour"),
    spark_round(spark_sum("precipitation"), 2).alias("total_precipitation_mm"),
    spark_sum("is_rain").alias("rainy_hours"),
    count("*").alias("total_hours")
).withColumn(
    "rain_percentage",
    spark_round((col("rainy_hours") / col("total_hours") * 100), 2)
).orderBy("year", "month")

print(f"   Computed {monthly_trends.count()} months")
monthly_trends.show(5)

monthly_trends.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="monthly_trends", keyspace="weather") \
    .save()
print("   ✓ Saved to Cassandra")

monthly_trends.unpersist()
del monthly_trends
print("   ✓ Memory cleared\n")

print("[2/6] Computing and saving seasonal trends...")

df_with_season = df.withColumn(
    "season",
    when((col("month") >= 3) & (col("month") <= 5), "Spring")
    .when((col("month") >= 6) & (col("month") <= 8), "Summer")
    .when((col("month") >= 9) & (col("month") <= 11), "Fall")
    .otherwise("Winter")
)

seasonal_trends = df_with_season.groupBy("year", "season").agg(
    spark_round(avg("temperature_2m"), 2).alias("avg_temperature"),
    spark_round(spark_min("temperature_2m"), 2).alias("min_temperature"),
    spark_round(spark_max("temperature_2m"), 2).alias("max_temperature"),
    spark_round(spark_sum("rain"), 2).alias("total_rain_mm"),
    spark_sum("is_rain").alias("rainy_hours"),
    count("*").alias("total_hours")
).withColumn(
    "rain_percentage",
    spark_round((col("rainy_hours") / col("total_hours") * 100), 2)
).orderBy("year", "season")

print(f"   Computed {seasonal_trends.count()} seasons")
seasonal_trends.show(5)

seasonal_trends.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="seasonal_trends", keyspace="weather") \
    .save()
print("   ✓ Saved to Cassandra")

seasonal_trends.unpersist()
del seasonal_trends
del df_with_season
print("   ✓ Memory cleared\n")

print("[3/6] Computing and saving hourly patterns...")

hourly_patterns = df.withColumn("hour_of_day", hour("timestamp")).groupBy("hour_of_day").agg(
    spark_round(avg("temperature_2m"), 2).alias("avg_temperature"),
    spark_round(avg("rain"), 4).alias("avg_rain"),
    spark_round(avg("precipitation"), 4).alias("avg_precipitation"),
    spark_sum("is_rain").alias("rainy_hours"),
    count("*").alias("total_hours")
).withColumn(
    "rain_probability",
    spark_round((col("rainy_hours") / col("total_hours") * 100), 2)
).orderBy("hour_of_day")

print(f"   Computed {hourly_patterns.count()} hours")
hourly_patterns.show(5)

hourly_patterns.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="hourly_patterns", keyspace="weather") \
    .save()
print("   ✓ Saved to Cassandra")

hourly_patterns.unpersist()
del hourly_patterns
print("   ✓ Memory cleared\n")

print("[4/6] Computing and saving day of week patterns...")

dow_patterns = df.withColumn("day_of_week", dayofweek("timestamp")).groupBy("day_of_week").agg(
    spark_round(avg("temperature_2m"), 2).alias("avg_temperature"),
    spark_round(avg("rain"), 4).alias("avg_rain"),
    spark_sum("is_rain").alias("rainy_hours"),
    count("*").alias("total_hours")
).withColumn(
    "rain_probability",
    spark_round((col("rainy_hours") / col("total_hours") * 100), 2)
).withColumn(
    "day_name",
    when(col("day_of_week") == 1, "Sunday")
    .when(col("day_of_week") == 2, "Monday")
    .when(col("day_of_week") == 3, "Tuesday")
    .when(col("day_of_week") == 4, "Wednesday")
    .when(col("day_of_week") == 5, "Thursday")
    .when(col("day_of_week") == 6, "Friday")
    .when(col("day_of_week") == 7, "Saturday")
).orderBy("day_of_week")

print(f"   Computed {dow_patterns.count()} days")
dow_patterns.show()

dow_patterns.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="dow_patterns", keyspace="weather") \
    .save()
print("   ✓ Saved to Cassandra")

dow_patterns.unpersist()
del dow_patterns
print("   ✓ Memory cleared\n")

print("[5/6] Computing and saving temperature extremes...")

temp_extremes = df.groupBy("year", "month").agg(
    spark_round(spark_min("temperature_2m"), 2).alias("coldest_temp"),
    spark_round(spark_max("temperature_2m"), 2).alias("hottest_temp")
).withColumn(
    "temp_range",
    spark_round(col("hottest_temp") - col("coldest_temp"), 2)
).orderBy("year", "month")

print(f"   Computed {temp_extremes.count()} months")
temp_extremes.show(5)

temp_extremes.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="temp_extremes", keyspace="weather") \
    .save()
print("   ✓ Saved to Cassandra")

temp_extremes.unpersist()
del temp_extremes
print("   ✓ Memory cleared\n")

print("[6/6] Computing and saving overall summary...")

overall_summary = df.agg(
    spark_round(avg("temperature_2m"), 2).alias("avg_temperature"),
    spark_round(spark_min("temperature_2m"), 2).alias("min_temperature"),
    spark_round(spark_max("temperature_2m"), 2).alias("max_temperature"),
    spark_round(stddev("temperature_2m"), 2).alias("stddev_temperature"),
    spark_round(spark_sum("rain"), 2).alias("total_rain_mm"),
    spark_round(avg("rain"), 4).alias("avg_rain_per_hour"),
    spark_round(spark_max("rain"), 2).alias("max_rain_hour"),
    spark_sum("is_rain").alias("total_rainy_hours"),
    count("*").alias("total_hours")
).withColumn(
    "rain_percentage",
    spark_round((col("total_rainy_hours") / col("total_hours") * 100), 2)
).withColumn(
    "id",
    lit(1)
)

print("   Overall statistics computed")
overall_summary.show()

overall_summary.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("append") \
    .options(table="overall_summary", keyspace="weather") \
    .save()
print("   ✓ Saved to Cassandra")

overall_summary.unpersist()
del overall_summary
print("   ✓ Memory cleared\n")

print("\n" + "=" * 80)
print("CASSANDRA TABLE SCHEMAS")
print("=" * 80)
print("\nRun these CQL commands in Cassandra if tables don't exist:\n")

cassandra_schemas = """
CREATE TABLE IF NOT EXISTS weather.monthly_trends (
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

CREATE TABLE IF NOT EXISTS weather.seasonal_trends (
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

CREATE TABLE IF NOT EXISTS weather.hourly_patterns (
    hour_of_day INT PRIMARY KEY,
    avg_temperature DOUBLE,
    avg_rain DOUBLE,
    avg_precipitation DOUBLE,
    rainy_hours BIGINT,
    total_hours BIGINT,
    rain_probability DOUBLE
);

CREATE TABLE IF NOT EXISTS weather.dow_patterns (
    day_of_week INT PRIMARY KEY,
    day_name TEXT,
    avg_temperature DOUBLE,
    avg_rain DOUBLE,
    rainy_hours BIGINT,
    total_hours BIGINT,
    rain_probability DOUBLE
);
"""

print(cassandra_schemas)

print("\n" + "=" * 80)
print("TREND ANALYSIS COMPLETE ✓")
print("=" * 80)
print(f"\nAnalyzed {total_records:,} records")
print(f"Date range: {date_stats['start_date']} to {date_stats['end_date']}")
print("\nTrends saved to Cassandra tables:")
print("  ✓ weather.monthly_trends - Monthly aggregations")
print("  ✓ weather.seasonal_trends - Seasonal patterns")
print("  ✓ weather.hourly_patterns - Hourly patterns (0-23)")
print("  ✓ weather.dow_patterns - Day of week patterns")
print("  ✓ weather.temp_extremes - Temperature extremes by month")
print("  ✓ weather.overall_summary - Overall statistics")
print("=" * 80)

spark.stop()
