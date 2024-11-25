# Databricks notebook source
# MAGIC %md
# MAGIC %md
# MAGIC ### ✨ **Author Information** 
# MAGIC
# MAGIC |                    |                                                                 |
# MAGIC |--------------------|-----------------------------------------------------------------|
# MAGIC | **Author**         | Neelam Patidar                                                  |
# MAGIC | **Date**           | August 01, 2024                                                 |
# MAGIC | **Supervisor**     | Prof. Jongwook Woo                                              |
# MAGIC | **Affiliation**    | Big Data AI Center (BigDAI): High Performance Information Computing Center (HiPIC) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 📚 **Tutorial Overview**
# MAGIC
# MAGIC In this sample you will use pyspark that analyzes website log files to get insight into how
# MAGIC customers use the website. With this analysis, you can see the frequency of visits to the website in a day
# MAGIC from external websites, and a summary of website errors that the users experience.
# MAGIC
# MAGIC In this tutorial, you'll learn how to use DBFS(Databricks File System) to:
# MAGIC - Upload the website log files to databricks & change it to dataframe for data manipulation
# MAGIC - Create tables to query those logs
# MAGIC - Create queries to analyze the data


from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import col, concat_ws, regexp_replace, countDistinct
from pyspark.sql.functions import year, month, dayofmonth, col, countDistinct
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.context import SparkContext

# Create or retrieve a Spark session
spark = SparkSession.builder.appName("Log File Analysis").getOrCreate()

# Assuming you have already started Spark session as 'spark'
weblog_file_location = "/user/npatida/SampleLog/909f2b.log"
file_type = "log"

# Read the log file as text
df_web_logs = spark.read.text(weblog_file_location)

# Skip the first two rows
df_skipped = df_web_logs.rdd.zipWithIndex().filter(lambda x: x[1] > 1).map(lambda x: x[0])

# Convert back to DataFrame
df_weblogs = spark.createDataFrame(df_skipped, df_web_logs.schema)
df_weblogs.show()

# Define the schema based on the sample data provided
weblog_schema = StructType([
    StructField("date", StringType(), True),
    StructField("time", StringType(), True),
    StructField("s_sitename", StringType(), True),
    StructField("cs_method", StringType(), True),
    StructField("cs_uristem", StringType(), True),
    StructField("cs_uriquery", StringType(), True),
    StructField("s_port", IntegerType(), True),
    StructField("cs_username", StringType(), True),
    StructField("c_ip", StringType(), True),
    StructField("cs_useragent", StringType(), True),
    StructField("cs_cookie", StringType(), True),
    StructField("cs_referer", StringType(), True),
    StructField("cs_host", StringType(), True),
    StructField("sc_status", IntegerType(), True),
    StructField("sc_substatus", IntegerType(), True),
    StructField("sc_win32status", IntegerType(), True),
    StructField("sc_bytes", IntegerType(), True),
    StructField("cs_bytes", IntegerType(), True),
    StructField("time_taken", IntegerType(), True)
])

# Assuming SparkSession is already created and named as 'spark'
# Load the data using the corrected delimiter and schema
df_weblogs = spark.read.format('csv') \
    .option("header", "true") \
    .option("delimiter", " ") \
    .schema(weblog_schema) \
    .load("/user/npatida/SampleLog/909f2b.log")


# Add a sequential index to each row
df_weblogs_with_index = df_weblogs.withColumn("index", monotonically_increasing_id())

# Filter out the first two rows based on index
df_weblogs = df_weblogs_with_index.filter("index > 1").drop("index")

# Display the DataFrame
df_weblogs.show()

#printSchema
df_weblogs.printSchema()

# Limit the DataFrame to 10 rows
df_weblogs_limited = df_weblogs.limit(10)

# Use the display function to show the DataFrame in Databricks
df_weblogs_limited.show()

# Filter and transform the data as per the requirements. Filter & Where are same.
df_client_errors = df_weblogs.filter((col("sc_status") >= 400) & (col("sc_status") < 500)) \
    .withColumn(
        "cs_page",
        concat_ws('?', col("cs_uristem"), regexp_replace(col("cs_uriquery"), 'X-ARR-LOGID=[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', ''))
    ) \
    .groupBy("sc_status", "cs_referer", "cs_page") \
    .agg(countDistinct("c_ip").alias("cnt")) \
    .orderBy(col("cnt").desc())

# Select top 10 results based on 'cnt'
df_top_10 = df_client_errors.orderBy(col("cnt").desc()).limit(10)
df_top_10.show()

# Filter and transform the data as per the requirements
df_refers_per_day = df_weblogs.filter((col("sc_status") >= 200) & (col("sc_status") < 300)) \
    .withColumn("year", year(col("date"))) \
    .withColumn("month", month(col("date"))) \
    .withColumn("day", dayofmonth(col("date"))) \
    .groupBy("year", "month", "day", "cs_referer") \
    .agg(countDistinct("c_ip").alias("cnt")) \
    .orderBy(col("cnt").desc())

df_refers_per_day.show()

# Select top 10 results based on 'cnt'
df_top_10 = df_refers_per_day.select("year", "month","Day", "cs_referer","cnt").limit(10)
df_top_10.show()









