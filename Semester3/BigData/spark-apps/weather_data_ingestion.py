from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, year, month, dayofmonth, hour, to_timestamp, round as spark_round
from pyspark.sql.functions import lag, avg as spark_avg, sum as spark_sum
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

spark = SparkSession.builder \
    .appName("Weather Data Ingestion") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

schema = StructType([
    StructField("date", StringType(), True),
    StructField("temperature_2m", DoubleType(), True),
    StructField("rain", DoubleType(), True),
    StructField("precipitation", DoubleType(), True),
])

print("Reading CSV data...")
df = spark.read.csv(
    "file:///data/historical_weather_data.csv",
    sep=",",
    header=True,
    schema=schema,
    mode="PERMISSIVE"
)

print(f"Initial record count: {df.count()}")

print("Cleaning data...")

df = df.withColumn("timestamp", to_timestamp(col("date"))) \
       .withColumn("year", year(col("timestamp"))) \
       .withColumn("month", month(col("timestamp"))) \
       .withColumn("day", dayofmonth(col("timestamp"))) \
       .withColumn("hour", hour(col("timestamp")))

df = df.withColumn("is_rain",
                   when((col("rain") > 0) | (col("precipitation") > 0), 1).otherwise(0))

df = df.fillna({"rain": 0.0, "precipitation": 0.0})
df = df.dropna(subset=["temperature_2m", "timestamp"])
df = df.dropDuplicates(["timestamp"])

df = df.filter(
    (col("temperature_2m") > -30) & (col("temperature_2m") < 40) &
    (col("rain") >= 0) & (col("rain") < 200) &
    (col("precipitation") >= 0) & (col("precipitation") < 200)
)

df = df.withColumn("temperature_2m", spark_round(col("temperature_2m"), 2)) \
       .withColumn("rain", spark_round(col("rain"), 2)) \
       .withColumn("precipitation", spark_round(col("precipitation"), 2))

window_spec = Window.orderBy("timestamp")
df = df.withColumn("temp_prev_hour", lag("temperature_2m", 1).over(window_spec))
df = df.withColumn("temp_change",
                   when(col("temp_prev_hour").isNotNull(),
                        col("temperature_2m") - col("temp_prev_hour")).otherwise(0))

window_3h = Window.orderBy("timestamp").rowsBetween(-3, 0)
df = df.withColumn("temp_3h_avg", spark_avg("temperature_2m").over(window_3h))
df = df.withColumn("rain_3h_sum", spark_sum("rain").over(window_3h))

print(f"Cleaned record count: {df.count()}")

print("Writing to HDFS...")
HDFS_OUTPUT_PATH = "hdfs://namenode:9000/user/hadoop/weather_prediction/data"

df.write \
    .mode("overwrite") \
    .partitionBy("year", "month") \
    .option("compression", "snappy") \
    .parquet(f"{HDFS_OUTPUT_PATH}/historical_weather")

print(f"Data written to: {HDFS_OUTPUT_PATH}/historical_weather")
print("Ingestion complete!")

spark.stop()
