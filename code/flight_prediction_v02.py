
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import re
import csv
from datetime import datetime, timedelta

from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier

# libraries for the classification sections 3A,3B,3B
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pandas as pd


# This is the important part for the user
# Set up the correct path to read the file

path = "gs://data-flights/flights_sample_3m.csv"

# Initialize Spark
spark = SparkSession.builder.appName("Flight").getOrCreate()

sc = spark.sparkContext

# Load file line by line
flight_rdd = sc.textFile(path)

#test_lines = sc.textFile(test_path)


# Shuffle + split
dev_rdd  = flight_rdd

#dev_rdd, main_rdd = flight_rdd.randomSplit([0.7, 0.3], seed=42)

#print("Dev size:", dev_rdd.count())
#print("Main size:", main_rdd.count())

for line in dev_rdd.take(5):
    print(line)

header = dev_rdd.first()

rows = dev_rdd.filter(lambda x: x != header) \
              .map(lambda x: next(csv.reader([x])))


# Pre-processing
def is_holiday(date_str):
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        year = d.year
        
        # Helper to find nth weekday of a month
        def nth_weekday(year, month, weekday, n):
            first = datetime(year, month, 1)
            first_weekday = first.weekday()
            delta = (weekday - first_weekday + 7) % 7
            return first + timedelta(days=delta + 7*(n-1))

        # Helper to find last weekday of a month
        def last_weekday(year, month, weekday):
            if month == 12:
                next_month = datetime(year+1, 1, 1)
            else:
                next_month = datetime(year, month+1, 1)
            last_day = next_month - timedelta(days=1)
            last_weekday = last_day.weekday()
            delta = (last_weekday - weekday + 7) % 7
            return last_day - timedelta(days=delta)

        # Fixed holidays
        holidays = {
            (1, 1),    # New Year's Day
            (6, 19),   # Juneteenth
            (7, 4),    # Independence Day
            (11, 11),  # Veterans Day
            (12, 25),  # Christmas
        }
        
        # Check fixed-date holidays
        if (d.month, d.day) in holidays:
            return 1

        # MLK Day: 3rd Monday of January
        if d.date() == nth_weekday(year, 1, 0, 3).date():
            return 1
        
        # Presidents Day: 3rd Monday of February
        if d.date() == nth_weekday(year, 2, 0, 3).date():
            return 1
        
        # Memorial Day: last Monday of May
        if d.date() == last_weekday(year, 5, 0).date():
            return 1
        
        # Labor Day: 1st Monday of September
        if d.date() == nth_weekday(year, 9, 0, 1).date():
            return 1
        
        # Columbus Day: 2nd Monday of October
        if d.date() == nth_weekday(year, 10, 0, 2).date():
            return 1
        
        # Thanksgiving: 4th Thursday of November
        if d.date() == nth_weekday(year, 11, 3, 4).date():
            return 1

        return 0
    
    except:
        return 0

# Prepare delay RDD
delay_rdd = rows.map(lambda row: (
    row[6],                      # origin
    row[8],                      # dest
    float(row[12]) if row[12] else 0.0
))

# Average delay per origin
avg_origin_delay = (
    delay_rdd
    .map(lambda x: (x[0], (x[2], 1)))
    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))
    .mapValues(lambda x: x[0] / x[1])
    .collectAsMap()
)

# Average delay per route
avg_route_delay = (
    delay_rdd
    .map(lambda x: ((x[0], x[1]), (x[2], 1)))
    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))
    .mapValues(lambda x: x[0] / x[1])
    .collectAsMap()
)

# Broadcast
avg_origin_delay_b = sc.broadcast(avg_origin_delay)
avg_route_delay_b  = sc.broadcast(avg_route_delay)

# placeholder
avg_weather = 0.0  

def weather_flags(raw):
    try:
        x = float(raw) if raw else 0.0
        return (
            1.0 if x == 0 else 0.0,       # is_clear
            1.0 if 0 < x < 15 else 0.0,   # is_showers
            1.0 if 15 <= x < 40 else 0.0, # is_storm
            1.0 if x >= 40 else 0.0       # is_snow
        )
    except:
        return (1.0, 0.0, 0.0, 0.0)       # default clear

# 6. Row parsing with features
def safe_convert(row):
    try:
        fl_date = row[0]
        airline = row[3]
        origin  = row[6]
        dest    = row[8]

        crs_dep_time = float(row[10]) if row[10] else 0.0
        dep_delay    = float(row[12]) if row[12] else 0.0
        distance     = float(row[26]) if row[26] else 0.0
        
        delayed = 1 if dep_delay >= 15 else 0

        # Date components
        try:
            d = datetime.strptime(fl_date, "%Y-%m-%d")
            month = d.month
            day_of_week = d.weekday()
        except:
            month = 0
            day_of_week = 0

        # Historical averages
        hist_origin_delay = avg_origin_delay_b.value.get(origin, 0.0)
        hist_route_delay  = avg_route_delay_b.value.get((origin, dest), 0.0)

        holiday_flag = is_holiday(fl_date)
        
        # Weather flags
        is_clear, is_showers, is_storm, is_snow = weather_flags(row[28])

        return (
            fl_date, airline, origin, dest, crs_dep_time, distance,
            month, day_of_week, hist_origin_delay, hist_route_delay,
            holiday_flag, is_clear, is_showers, is_storm, is_snow,
            delayed
        )

    except:
        return None

# Apply parsing
selected = rows.map(safe_convert).filter(lambda x: x is not None)
# end section 1


# Section 2
# Create DataFrame from RDD
columns = [
    "date", "airline", "origin", "dest",
    "crs_dep_time", "distance",
    "month", "day_of_week",
    "hist_origin_delay", "hist_route_delay",
    "holiday_flag", 
    "is_clear", "is_showers", "is_storm", "is_snow",
    "delayed"
]

df = spark.createDataFrame(selected, schema=columns)

# Drop rows with missing critical values
required_cols = [
    "airline", "origin", "dest",
    "crs_dep_time", "distance",
    "month", "day_of_week",
    "hist_origin_delay", "hist_route_delay",
    "holiday_flag", 
    "is_clear", "is_showers", "is_storm", "is_snow",
    "delayed"
]
df = df.dropna(subset=required_cols)

# Reduce airport cardinality safely
topN = 30
top_origins = [r["origin"] for r in df.repartition(10, "origin")
                                  .groupBy("origin").count()
                                  .orderBy(F.desc("count"))
                                  .limit(topN)
                                  .toLocalIterator()]

top_dests = [r["dest"] for r in df.repartition(10, "dest")
                                .groupBy("dest").count()
                                .orderBy(F.desc("count"))
                                .limit(topN)
                                .toLocalIterator()]

df = df.withColumn("origin", F.when(F.col("origin").isin(top_origins), F.col("origin")).otherwise("other"))
df = df.withColumn("dest",   F.when(F.col("dest").isin(top_dests), F.col("dest")).otherwise("other"))

# Cast numeric columns
numeric_cols = [
    "crs_dep_time", "distance",
    "month", "day_of_week",
    "hist_origin_delay", "hist_route_delay",
    "holiday_flag",
    "is_clear", "is_showers", "is_storm", "is_snow"
]

for c in numeric_cols + ["delayed"]:
    df = df.withColumn(c, F.col(c).cast(FloatType()))

# Time-of-day features
df = df.withColumn("hour", (F.col("crs_dep_time") / 100).cast("int"))
df = df.withColumn("hour_sin", F.sin(F.col("hour") * (2 * 3.14159265 / 24)))
df = df.withColumn("hour_cos", F.cos(F.col("hour") * (2 * 3.14159265 / 24)))
numeric_cols += ["hour_sin", "hour_cos"]

df = df.withColumn(
    "time_bin",
    F.when(F.col("hour") < 6, 0)
     .when(F.col("hour") < 12, 1)
     .when(F.col("hour") < 18, 2)
     .otherwise(3)
)

# Congestion features
origin_cong = df.groupBy("origin").count().withColumnRenamed("count","origin_congestion")
dest_cong   = df.groupBy("dest").count().withColumnRenamed("count","dest_congestion")

df = df.join(origin_cong, on="origin", how="left")
df = df.join(dest_cong,   on="dest",   how="left")

df = df.withColumn("origin_congestion", F.col("origin_congestion").cast(FloatType()))
df = df.withColumn("dest_congestion",   F.col("dest_congestion").cast(FloatType()))
numeric_cols += ["origin_congestion", "dest_congestion"]


# Train/test split
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# Handle class imbalance (oversample minority)
count_0 = train_df.filter("delayed = 0").count()
count_1 = train_df.filter("delayed = 1").count()

minority_class = 1.0 if count_1 < count_0 else 0.0
minority_df = train_df.filter(F.col("delayed") == minority_class)
majority_df = train_df.filter(F.col("delayed") != minority_class)

ratio = max(count_0, count_1) / min(count_0, count_1)
minority_oversampled = minority_df.sample(withReplacement=True, fraction=ratio, seed=42)
train_df_balanced = majority_df.union(minority_oversampled).cache()  # cache smaller DF
train_df_balanced.count()  # now safe to materialize

# Categorical features → StringIndexer
categorical_cols = ["airline", "origin", "dest", "time_bin"]
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="skip") for c in categorical_cols]

# VectorAssembler
assembler = VectorAssembler(
    inputCols=[f"{c}_idx" for c in categorical_cols] + numeric_cols,
    outputCol="features"
)

# end Section 2


# Section 3A - Evaluates linear regression for
# a small grid of values

# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="delayed", maxIter=100)

# Pipeline
pipeline = Pipeline(stages=indexers + [assembler, lr])

# Small hyperparameter grid
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
    .build()

# Evaluators
auc_evaluator = BinaryClassificationEvaluator(labelCol="delayed", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
acc_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="accuracy")
prec_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="precisionByLabel")
rec_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="recallByLabel")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="f1")

# Store results
results = []

# Loop over ParamMaps
for param_map in param_grid:
    # Create a fresh LR instance for each param_map
    lr_run = lr.copy(param_map)
    pipeline_run = Pipeline(stages=indexers + [assembler, lr_run])
    
    # Fit model
    model = pipeline_run.fit(train_df_balanced)
    
    # Predict
    predictions = model.transform(test_df)
    
    # Evaluate metrics
    results.append({
        "regParam": lr_run.getRegParam(),
        "elasticNetParam": lr_run.getElasticNetParam(),
        "AUC": auc_evaluator.evaluate(predictions),
        "Accuracy": acc_evaluator.evaluate(predictions),
        "Precision": prec_evaluator.evaluate(predictions),
        "Recall": rec_evaluator.evaluate(predictions),
        "F1": f1_evaluator.evaluate(predictions)
    })

# Convert results to Pandas DataFrame
results_df = pd.DataFrame(results)
print(results_df.sort_values("AUC", ascending=False))
# end Section 3A


# Section 3B - Evaluates random forest for
# a small grid of values

rf = RandomForestClassifier(featuresCol="features", labelCol="delayed", seed=42)

# Pipeline
pipeline = Pipeline(stages=indexers + [assembler, rf])

# Small hyperparameter grid
# To be narrow down to only the best 
# set of values and compare the
# performance with the
# other classifiers (LR,GBT)

param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

# Evaluators
auc_evaluator = BinaryClassificationEvaluator(labelCol="delayed", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
acc_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="accuracy")
prec_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="precisionByLabel")
rec_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="recallByLabel")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="f1")

# Store results
results = []

# Loop over ParamMaps
for param_map in param_grid:
    # Create a fresh RF instance for this ParamMap
    rf_run = rf.copy(param_map)
    pipeline_run = Pipeline(stages=indexers + [assembler, rf_run])
    
    # Fit model
    model = pipeline_run.fit(train_df_balanced)
    
    # Predict
    predictions = model.transform(test_df)
    
    # Evaluate metrics
    results.append({
        "numTrees": rf_run.getNumTrees(),
        "maxDepth": rf_run.getMaxDepth(),
        "AUC": auc_evaluator.evaluate(predictions),
        "Accuracy": acc_evaluator.evaluate(predictions),
        "Precision": prec_evaluator.evaluate(predictions),
        "Recall": rec_evaluator.evaluate(predictions),
        "F1": f1_evaluator.evaluate(predictions)
    })

# Convert results to Pandas DataFrame for comparison
results_df = pd.DataFrame(results)
print(results_df.sort_values("AUC", ascending=False))


# end Section 3B

# Section 3C - Grid for the GBT classifier

gbt = GBTClassifier(featuresCol="features", labelCol="delayed", seed=42)

# Pipeline
pipeline = Pipeline(stages=indexers + [assembler, gbt])

# Small hyperparameter grid
# To be narrow down to only the best 
# set of values and compare the
# performance with the
# other classifiers (RT,RF)

param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [50, 100]) \
    .addGrid(gbt.maxDepth, [3, 5]) \
    .build()

# Evaluators
auc_evaluator = BinaryClassificationEvaluator(labelCol="delayed", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
acc_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="accuracy")
prec_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="precisionByLabel")
rec_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="recallByLabel")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="f1")

# Store results
results = []

# Loop over ParamMaps
for param_map in param_grid:
    # Create a fresh GBT instance for this ParamMap
    gbt_run = gbt.copy(param_map)
    pipeline_run = Pipeline(stages=indexers + [assembler, gbt_run])
    
    # Fit model
    model = pipeline_run.fit(train_df_balanced)
    
    # Predict
    predictions = model.transform(test_df)
    
    # Evaluate metrics
    results.append({
        "maxIter": gbt_run.getMaxIter(),
        "maxDepth": gbt_run.getMaxDepth(),
        "AUC": auc_evaluator.evaluate(predictions),
        "Accuracy": acc_evaluator.evaluate(predictions),
        "Precision": prec_evaluator.evaluate(predictions),
        "Recall": rec_evaluator.evaluate(predictions),
        "F1": f1_evaluator.evaluate(predictions)
    })

# Convert results to Pandas DataFrame for comparison
results_df = pd.DataFrame(results)
print(results_df.sort_values("AUC", ascending=False))

# end section 3A




