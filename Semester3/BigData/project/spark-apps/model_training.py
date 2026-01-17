from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from xgboost.spark import SparkXGBRegressor
import time
import json

spark = SparkSession.builder \
    .appName("Rain Amount Prediction - XGBoost Regression") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.memory", "1g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 70)
print("RAIN AMOUNT PREDICTION - XGBOOST REGRESSION")
print("Predicting: rain (mm)")
print("=" * 70)

HDFS_INPUT_PATH = "hdfs://namenode:9000/user/hadoop/weather_prediction/data/historical_weather"
HDFS_MODEL_OUTPUT_PATH = "hdfs://namenode:9000/user/hadoop/weather_prediction/models/xgboost_rain_regression"
LOCAL_MODEL_OUTPUT_PATH = "/models/rain_prediction_regression"

print("\nReading data from HDFS...")
print(f"Input path: {HDFS_INPUT_PATH}")

df = spark.read.parquet(HDFS_INPUT_PATH)

print(f"Total records loaded: {df.count()}")
df.printSchema()

print("\nSample data:")
df.select("timestamp", "temperature_2m", "rain", "precipitation").show(10)

df = df.fillna(0, subset=["rain"])

print("\nRain statistics:")
df.select("rain").describe().show()

rain_records = df.filter(col("rain") > 0).count()
total_records = df.count()
print(f"Records with rain: {rain_records} ({rain_records/total_records*100:.2f}%)")

print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

feature_columns = [
    "temperature_2m",
    "month",
    "day",
    "hour"
]

if "relative_humidity_2m" in df.columns:
    feature_columns.append("relative_humidity_2m")
    df = df.fillna(df.agg({"relative_humidity_2m": "mean"}).first()[0], subset=["relative_humidity_2m"])
    print("✓ Added relative_humidity_2m")

if "surface_pressure" in df.columns:
    feature_columns.append("surface_pressure")
    df = df.fillna(df.agg({"surface_pressure": "mean"}).first()[0], subset=["surface_pressure"])
    print("✓ Added surface_pressure")

print(f"\nFeatures: {feature_columns}")
print(f"Target: rain")

df_clean = df.select(feature_columns + ["rain"]).na.drop()

print(f"\nRecords after removing nulls: {df_clean.count()}")

df_clean = df_clean.filter(col("rain") > 0)

print(f"Training only on rainy hours: {df_clean.count()}")


print("\nSplitting data...")
train_data, test_data = df_clean.randomSplit([0.8, 0.2], seed=42)

print(f"Training set size: {train_data.count()}")
print(f"Test set size: {test_data.count()}")

train_data.cache()
test_data.cache()

print("\n" + "=" * 70)
print("BUILDING XGBOOST REGRESSION PIPELINE")
print("=" * 70)

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

xgb = SparkXGBRegressor(
    features_col="features",
    label_col="rain",
    num_workers=2,
    max_depth=6,
    n_estimators=100,
    learning_rate=0.1,
    reg_alpha=0.0,
    reg_lambda=1.0,
    min_child_weight=1,
    gamma=0.0,
    subsample=0.8,
    colsample_bytree=0.8,
    seed=42
)

print(f"\nXGBoost Configuration:")
print(f"  Trees: {xgb.getOrDefault('n_estimators')}")
print(f"  Max Depth: {xgb.getOrDefault('max_depth')}")
print(f"  Learning Rate: {xgb.getOrDefault('learning_rate')}")

pipeline = Pipeline(stages=[assembler, xgb])

print("\n" + "=" * 70)
print("TRAINING XGBOOST REGRESSION MODEL")
print("=" * 70)
print("Training...")

start_time = time.time()
trained_model = pipeline.fit(train_data)
training_time = time.time() - start_time

print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

print("\n" + "=" * 70)
print("EVALUATING MODEL")
print("=" * 70)

predictions = trained_model.transform(test_data)

rmse_evaluator = RegressionEvaluator(
    labelCol="rain",
    predictionCol="prediction",
    metricName="rmse"
)
rmse = rmse_evaluator.evaluate(predictions)

mae_evaluator = RegressionEvaluator(
    labelCol="rain",
    predictionCol="prediction",
    metricName="mae"
)
mae = mae_evaluator.evaluate(predictions)

r2_evaluator = RegressionEvaluator(
    labelCol="rain",
    predictionCol="prediction",
    metricName="r2"
)
r2 = r2_evaluator.evaluate(predictions)

print(f"\nRegression Model Performance:")
print(f"  RMSE (Root Mean Squared Error): {rmse:.4f} mm")
print(f"  MAE (Mean Absolute Error):      {mae:.4f} mm")
print(f"  R² (Coefficient of Determination): {r2:.4f}")

print("\nSample Predictions:")
predictions.select("temperature_2m", "hour", "rain", "prediction").show(20)

print("\nPredictions for hours with actual rain:")
predictions.filter(col("rain") > 0).select("temperature_2m", "hour", "rain", "prediction").show(20)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

xgb_model = trained_model.stages[-1]
feature_importances = xgb_model.get_feature_importances()

print(f"\n{'Feature':<25} {'Importance':<15} {'Bar':<30}")
print("-" * 70)
for feature, importance in feature_importances.items():
    bar = "█" * int(importance * 100)
    print(f"{feature:<25} {importance:<15.4f} {bar}")

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

try:
    print(f"\nSaving to HDFS: {HDFS_MODEL_OUTPUT_PATH}")
    trained_model.write().overwrite().save(HDFS_MODEL_OUTPUT_PATH)
    print(f"✓ Model saved to HDFS")
except Exception as e:
    print(f"✗ Error saving to HDFS: {e}")

try:
    import os
    os.makedirs(LOCAL_MODEL_OUTPUT_PATH, exist_ok=True)
    print(f"\nSaving to local: {LOCAL_MODEL_OUTPUT_PATH}")
    trained_model.write().overwrite().save(LOCAL_MODEL_OUTPUT_PATH)
    print(f"✓ Model saved locally")
except Exception as e:
    print(f"✗ Error saving locally: {e}")

print("\nSaving model metadata...")
metadata = {
    "model_type": "XGBoost Regression",
    "target": "rain (mm)",
    "xgboost_params": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "seed": 42
    },
    "features": feature_columns,
    "feature_importance": feature_importances,
    "metrics": {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "Training_Time_Seconds": training_time
    },
    "training_info": {
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_records": train_data.count(),
        "test_records": test_data.count(),
        "rainy_records_percentage": float(rain_records/total_records*100)
    }
}

metadata_json = json.dumps(metadata, indent=2)
print("\nModel Metadata:")
print(metadata_json)

try:
    metadata_path = f"{HDFS_MODEL_OUTPUT_PATH}_metadata"
    metadata_df = spark.createDataFrame([(metadata_json,)], ["metadata"])
    metadata_df.coalesce(1).write.mode("overwrite").text(metadata_path)
    print(f"✓ Metadata saved to {metadata_path}")
except Exception as e:
    print(f"✗ Error saving metadata: {e}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE ✓")
print("=" * 70)
print(f"\nModel Type: XGBoost Regression")
print(f"Target: rain (mm)")
print(f"Features: {feature_columns}")
print(f"\nPerformance Metrics:")
print(f"  RMSE: {rmse:.4f} mm")
print(f"  MAE:  {mae:.4f} mm")
print(f"  R²:   {r2:.4f}")
print(f"\nTraining Info:")
print(f"  Training time: {training_time:.2f} seconds")
print(f"  Training records: {train_data.count():,}")
print(f"  Test records: {test_data.count():,}")
print(f"\nModel saved to:")
print(f"  HDFS:  {HDFS_MODEL_OUTPUT_PATH}")
print(f"  Local: {LOCAL_MODEL_OUTPUT_PATH}")
print("=" * 70)

train_data.unpersist()
test_data.unpersist()

spark.stop()
