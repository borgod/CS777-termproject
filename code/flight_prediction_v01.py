from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
import csv
from datetime import datetime

# ---------------------------
# Initialize Spark
# ---------------------------
spark = SparkSession.builder.appName("Flight").getOrCreate()
sc = spark.sparkContext

# ---------------------------
# Load CSV
# ---------------------------
path = "/Documents/yourfile.csv"
flight_rdd = sc.textFile(path)

# Split dev and main data
dev_rdd, main_rdd = flight_rdd.randomSplit([0.2, 0.8], seed=42)
print("Dev size:", dev_rdd.count())
print("Main size:", main_rdd.count())

# -- Section 1 ------
# CSV parsing

header = dev_rdd.first()
rows = dev_rdd.filter(lambda x: x != header).map(lambda x: next(csv.reader([x])))

# Holiday function
def is_holiday(date_str):
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return 1 if (d.month, d.day) in [(1,1),(7,4),(12,25)] else 0
    except:
        return 0

# Prepare historical delay features
delay_rdd = rows.map(lambda row: (row[6], row[8], float(row[12]) if row[12] else 0.0))

# Average delay per origin
avg_origin_delay = (
    delay_rdd
    .map(lambda x: (x[0], (x[2], 1)))
    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))
    .mapValues(lambda x: x[0]/x[1])
    .collectAsMap()
)

# Average delay per route
avg_route_delay = (
    delay_rdd
    .map(lambda x: ((x[0], x[1]), (x[2],1)))
    .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1]))
    .mapValues(lambda x: x[0]/x[1])
    .collectAsMap()
)

avg_origin_delay_b = sc.broadcast(avg_origin_delay)
avg_route_delay_b = sc.broadcast(avg_route_delay)
avg_weather = 0.0  # placeholder

# Parse rows with features
def safe_convert(row):
    try:
        fl_date = row[0]
        airline = row[3]
        origin = row[6]
        dest = row[8]
        crs_dep_time = float(row[10]) if row[10] else 0.0
        dep_delay = float(row[12]) if row[12] else 0.0
        distance = float(row[26]) if row[26] else 0.0
        delayed = 1 if dep_delay >= 15 else 0

        try:
            d = datetime.strptime(fl_date, "%Y-%m-%d")
            month = d.month
            day_of_week = d.weekday()
        except:
            month = 0
            day_of_week = 0

        hist_origin_delay = avg_origin_delay_b.value.get(origin, 0.0)
        hist_route_delay  = avg_route_delay_b.value.get((origin,dest), 0.0)
        holiday_flag = is_holiday(fl_date)
        weather_forecast = avg_weather

        return (
            fl_date, airline, origin, dest, crs_dep_time, distance,
            month, day_of_week, hist_origin_delay, hist_route_delay,
            holiday_flag, weather_forecast, delayed
        )
    except:
        return None

selected = rows.map(safe_convert).filter(lambda x: x is not None)


# End of Section 1
# -----------------


# --Section 2-----
# Create DataFrame
columns = [
    "date","airline","origin","dest","crs_dep_time","distance",
    "month","day_of_week","hist_origin_delay","hist_route_delay",
    "holiday_flag","weather_forecast","delayed"
]
df = spark.createDataFrame(selected, schema=columns)
df = df.dropna()

# Reduce cardinality of airports
# to reduce memory
topN = 30
top_origins = [r["origin"] for r in df.groupBy("origin").count().orderBy(F.desc("count")).limit(topN).collect()]
top_dests = [r["dest"] for r in df.groupBy("dest").count().orderBy(F.desc("count")).limit(topN).collect()]

df = df.withColumn("origin", F.when(F.col("origin").isin(top_origins), F.col("origin")).otherwise("other"))
df = df.withColumn("dest", F.when(F.col("dest").isin(top_dests), F.col("dest")).otherwise("other"))

# Cast numeric features
numeric_cols = ["crs_dep_time","distance","month","day_of_week",
                "hist_origin_delay","hist_route_delay","holiday_flag","weather_forecast"]
for c in numeric_cols + ["delayed"]:
    df = df.withColumn(c, F.col(c).cast(FloatType()))

# Categorical encoding
categorical_cols = ["airline","origin","dest"]
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in categorical_cols]

# End of section 2
# -----------------


# --- Section 3 ----
# Assemble features
assembler = VectorAssembler(inputCols=[f"{c}_ohe" for c in categorical_cols] + numeric_cols, outputCol="features")

# Logistic regression
lr = LogisticRegression(featuresCol="features", labelCol="delayed", maxIter=20)

pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

# Train/test split
train_df, test_df = df.randomSplit([0.7,0.3], seed=42)
model = pipeline.fit(train_df)

# Predictions & probability
preds = model.transform(test_df)
preds = preds.withColumn("prob_delay", F.round(vector_to_array("probability")[1],2)) \
             .select("airline","origin","dest","prob_delay","prediction","delayed")

preds.show(20, False)

# AUC
evaluator = BinaryClassificationEvaluator(labelCol="delayed", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(model.transform(test_df))
print("AUC =", round(auc,4))

# Confusion matrix
cm = preds.groupBy("delayed","prediction").count().orderBy("delayed","prediction")
cm.show()

# End of Section 3
# ----------------