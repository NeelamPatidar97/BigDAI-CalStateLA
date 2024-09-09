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
# MAGIC
# MAGIC **Prerequisites**:
# MAGIC - Access to a Databricks workspace
# MAGIC - Basic knowledge of Apache Spark and PySpark
# MAGIC - Log file in .log format (e.g., 909f2b.log)

# COMMAND ----------

# MAGIC %md
# MAGIC **Launch a Spark Cluster**: 
# MAGIC Configure a new Spark cluster in the Databricks dashboard ensuring compatibility with the needed Spark version.
# MAGIC
# MAGIC **Upload Data to DBFS**:
# MAGIC - Navigate to "Compute" -> "Create Table" in the UI to upload the [909f2b.log](https://github.com/dalgual/aidatasci/raw/master/data/bigdata/909f2b.log) file to the Databricks File System (DBFS)
# MAGIC - First, ensure your log file (909f2b.log) is uploaded to the DBFS. 
# MAGIC - Then, read the Log File in PySpark. Since the file is a log file, you'll initially read it as text to examine its structure. 
# MAGIC
# MAGIC Here’s how you can start by reading the file:

# COMMAND ----------

from pyspark.sql import SparkSession

# Assuming you have already started Spark session as 'spark'
weblog_file_location = "/FileStore/tables/909f2b-3.log"
file_type = "log"

# Read the log file as text
df_web_logs = spark.read.text(weblog_file_location)

# Skip the first two rows
df_skipped = df_web_logs.rdd.zipWithIndex().filter(lambda x: x[1] > 1).map(lambda x: x[0])

# Convert back to DataFrame
df_weblogs = spark.createDataFrame(df_skipped, df_web_logs.schema)
display(df_weblogs)


# COMMAND ----------

# MAGIC %md
# MAGIC #####Define and Apply Schema
# MAGIC Define a "weblog_schema" that reflects the structure of your log data to facilitate more efficient querying. After that, load the data using correct delimiter and schema. Then, display the result for df_weblogs dataframe.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

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
    .load("/FileStore/tables/909f2b-3.log")

# Add a sequential index to each row
df_weblogs_with_index = df_weblogs.withColumn("index", monotonically_increasing_id())

# Filter out the first two rows based on index
df_weblogs = df_weblogs_with_index.filter("index > 1").drop("index")

# Display the DataFrame
display(df_weblogs)


# COMMAND ----------

# MAGIC %md
# MAGIC Print the schema for "**df_weblogs**" DataFrame to understand the structure of your data, which is essential for subsequent data manipulation.

# COMMAND ----------

df_weblogs.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Limit the data to the first 10 rows to simplify initial data handling for **df_weblogs** and store it in new dataframe named "**df_weblogs_limited**"

# COMMAND ----------

# Limit the DataFrame to 10 rows
df_weblogs_limited = df_weblogs.limit(10)

# Use the display function to show the DataFrame in Databricks
display(df_weblogs_limited)

# COMMAND ----------

# MAGIC %md
# MAGIC The analysis for clienterrors extracts data from the weblogs table for HTTP status codes between 400 and
# MAGIC 500, and groups them by the users facing those errors and the type of error codes. The range of status
# MAGIC code between 400 and 500, represented by sc_status column in the weblogs table, corresponds to the
# MAGIC errors clients get while accessing the website. The extracted data is then sorted on the number of
# MAGIC occurrences of each error code and written to the **df_client_errors** table.
# MAGIC
# MAGIC **Filtering and Transforming Data**
# MAGIC
# MAGIC The goal is to identify client-side errors (HTTP status codes 400-499), extract meaningful information, and count distinct IPs to see how widespread the issues are.

# COMMAND ----------

from pyspark.sql.functions import col, concat_ws, regexp_replace, countDistinct

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
display(df_top_10)


# COMMAND ----------

# MAGIC %md
# MAGIC The query for refersperday extracts data from the weblogs table for all external websites referencing
# MAGIC this website. The external website information is extracted from the **cs_referer** column of **weblogs table**.
# MAGIC To make sure the referring links did not encounter an error, the table only shows data for pages that
# MAGIC returned an HTTP status code between 200 and 300. The extracted data is then written to
# MAGIC the **df_refers_per_day** table.

# COMMAND ----------

from pyspark.sql.functions import year, month, dayofmonth, col, countDistinct

# Filter and transform the data as per the requirements
df_refers_per_day = df_weblogs.filter((col("sc_status") >= 200) & (col("sc_status") < 300)) \
    .withColumn("year", year(col("date"))) \
    .withColumn("month", month(col("date"))) \
    .withColumn("day", dayofmonth(col("date"))) \
    .groupBy("year", "month", "day", "cs_referer") \
    .agg(countDistinct("c_ip").alias("cnt")) \
    .orderBy(col("cnt").desc())

display(df_refers_per_day)



# COMMAND ----------

# Select top 10 results based on 'cnt'
df_top_10 = df_refers_per_day.select("year", "month","Day", "cs_referer","cnt").limit(10)
display(df_top_10)
