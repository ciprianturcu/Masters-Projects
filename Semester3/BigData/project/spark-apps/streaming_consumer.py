from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, udf, lit, year, month, dayofmonth, hour
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
    .appName("Weather Streaming Consumer") \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.cassandra.connection.port", "9042") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

weather_schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("temperature", DoubleType(), True),
    StructField("humidity", DoubleType(), True),
    StructField("pressure", DoubleType(), True)
])

print("=" * 70)
print("Weather Streaming Consumer (XGBoost Integrated)")
print("=" * 70)

MODEL_PATH = "hdfs://namenode:9000/user/hadoop/weather_prediction/models/xgboost"
print(f"Loading model from: {MODEL_PATH}...")
try:
    model = PipelineModel.load(MODEL_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("CRITICAL: You must run model_training.py first!")
    import sys
    sys.exit(1)

print("\nReading stream from Kafka...")
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "weather-data") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

weather_df = kafka_df.select(
    from_json(col("value").cast("string"), weather_schema).alias("data")
).select("data.*")

weather_df = weather_df.withColumn("timestamp", to_timestamp(col("timestamp")))

def prepare_features(df):
    df = df.withColumnRenamed("temperature", "temperature_2m")

    df = df.withColumn("month", month(col("timestamp"))) \
           .withColumn("day", dayofmonth(col("timestamp"))) \
           .withColumn("hour", hour(col("timestamp")))

    df = df.withColumn("temp_change", lit(0.0)) \
           .withColumn("temp_3h_avg", col("temperature_2m")) \
           .withColumn("rain_3h_sum", lit(0.0))

    return df

def process_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        return

    print(f"\n--- Processing Batch {batch_id} ---")

    batch_df.cache()

    try:
        prediction_df = prepare_features(batch_df)

        predictions = model.transform(prediction_df)
        print(predictions.head(3))

        final_df = predictions.withColumn(
            "predicted_rain",
            col("prediction")
        )

        print("Predictions generated:")
        final_df.select("timestamp", "temperature_2m", "predicted_rain").show()

        batch_df.select(
            "timestamp", "temperature", "humidity", "pressure"
        ).write \
            .format("org.apache.spark.sql.cassandra") \
            .mode("append") \
            .options(table="weather_realtime", keyspace="weather") \
            .save()
        print("✓ Saved to weather_realtime")

        print()
        final_df.select(
            col("timestamp"),
            col("temperature_2m").alias("temperature"),
            col("humidity"),
            col("pressure"),
            col("predicted_rain")
        ).write \
            .format("org.apache.spark.sql.cassandra") \
            .mode("append") \
            .options(table="rain_predictions", keyspace="weather") \
            .save()
        print("✓ Saved to rain_predictions")

    except Exception as e:
        print(f"✗ Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        batch_df.unpersist()

print("\nStarting streaming query...")
query = weather_df.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/checkpoint/weather_streaming_xgboost") \
    .trigger(processingTime="10 seconds") \
    .start()

query.awaitTermination()
