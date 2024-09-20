# Databricks notebook source
# MAGIC %md
# MAGIC ### ✨ **Author Information** 
# MAGIC
# MAGIC |                    |                                                                 |
# MAGIC |--------------------|-----------------------------------------------------------------|
# MAGIC | **Author**         | Neelam Patidar                                                  |
# MAGIC | **Date**           | September 15, 2024                                                 |
# MAGIC | **Supervisor**     | Prof. Jongwook Woo                                              |
# MAGIC | **Affiliation**    | Big Data AI Center (BigDAI): High Performance Information Computing Center (HiPIC) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 📚 **Tutorial Overview**
# MAGIC
# MAGIC The data set here is twitter data about **Go game of Google’s Aphago vs Se-Dol Lee**, who is a Go
# MAGIC champion. The game was held in March 12 – March 17 2016. Alphago won 4 out of 5 games – Se-Dol
# MAGIC won the second game - so that Korean people and Go community were so shocked because they
# MAGIC believed that Artificial Intelligence cannot beat human as Go is more complicated than Chess even
# MAGIC though IBM Deep Blue defeated the chess champion in 1997. Since then, Korea and other Asian
# MAGIC countries studied and have been focusing on AI industry.

# COMMAND ----------

# MAGIC %md
# MAGIC **Load CSV Data into a DataFrame**
# MAGIC
# MAGIC First, load the CSV file into a DataFrame from the specified location in DBFS.

# COMMAND ----------

# File location and type
tweets_alphago_location = "/FileStore/tables/tweets_alphago.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
tweetsbi_df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(tweets_alphago_location)

display(tweetsbi_df.limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC **Create a Directory in DBFS**
# MAGIC
# MAGIC Create a new directory in DBFS to store the file.

# COMMAND ----------

# Create a directory in DBFS
dbutils.fs.mkdirs("/tmp/tweetsbi")


# COMMAND ----------

# MAGIC %md
# MAGIC **Copy the File to the New Directory**
# MAGIC
# MAGIC Move the file to the newly created directory for better organization.

# COMMAND ----------

# Copy the file to the newly created directory
file_path = "/FileStore/tables/tweets_alphago.csv"  # Update this path if the file location differs
destination_path = "/tmp/tweetsbi/tweets_alphago.csv"

# Move the file to the desired location
dbutils.fs.cp(file_path, destination_path)


# COMMAND ----------

# MAGIC %md
# MAGIC **List the Files in the Directory**
# MAGIC
# MAGIC Verify that the file has been successfully uploaded.

# COMMAND ----------

# List the files in the directory to verify the upload
files = dbutils.fs.ls("/tmp/tweetsbi")
for file in files:
    print(file.name, file.size)


# COMMAND ----------

# MAGIC %md
# MAGIC **Read the CSV File from DBFS**
# MAGIC
# MAGIC Read the CSV file from the new location into a DataFrame.

# COMMAND ----------

# Read the CSV file from DBFS
tweetsbi_df = spark.read.option("header", "true").csv("/tmp/tweetsbi/tweets_alphago.csv")

# Show the last 3 rows of the DataFrame
tweetsbi_df.tail(3)


# COMMAND ----------

# MAGIC %md
# MAGIC **Display the Schema of the DataFrame**
# MAGIC
# MAGIC Understand the structure of the DataFrame by displaying its schema.

# COMMAND ----------

# Display the schema to understand the structure of the DataFrame
tweetsbi_df.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC **Filter and Aggregate Data**
# MAGIC
# MAGIC Filter out invalid values and aggregate the data by country and sentiment.

# COMMAND ----------

from pyspark.sql.functions import col, count

# Filter out null, empty, and invalid values for country
Filtered_tweetsbi_df = tweetsbi_df.filter(
    (col('country').isNotNull()) &
    (col('country') != '') &
    (col('country') != 'null')
)

"""
Filtered_tweetsbi_df = tweetsbi_df.filter(
    'country is not null AND country != "" AND country != "null"')
"""

# Group by country and sentiment, and count occurrences
tweets_top_countries = Filtered_tweetsbi_df.groupBy('country', 'sentiment').agg(
    count('sentiment').alias('cnt')
).orderBy(col('cnt').desc())

# Display the results
tweets_top_countries.show(2)

#HIVEQL has different values
"""
JAPAN | 1 | 8066|
UNITED STATES | 1 | 4251
"""

# COMMAND ----------

# MAGIC %md
# MAGIC **Aggregate Data to Find Top 10 Countries**
# MAGIC
# MAGIC Find the top 10 countries based on sentiment counts.

# COMMAND ----------

# Group by country and sum the counts to create the top10 equivalent
top10_df = tweets_top_countries.groupBy("country") \
    .sum("cnt") \
    .withColumnRenamed("sum(cnt)", "cnt2") \
    .orderBy("cnt2", ascending=False) \
    .limit(10)

# Display the top 5 rows from the grouped DataFrame
top10_df.show(5)

#HIVEQL has different values
"""
UNITED STATES | 9360 |
JAPAN | 9252 |
RUSSIAN FEDERATION | 1623 |
KOREA (S) | 1498 |
UNITED KINGDOM | 1311
"""

# COMMAND ----------

# MAGIC %md
# MAGIC **Perform a Left Semi Join**
# MAGIC
# MAGIC Filter the *tweets_top_countries* DataFrame to include only the top 10 countries.

# COMMAND ----------

# Perform a left semi join to match the SQL join operation
tweets_top10_countries_df = tweets_top_countries.join(
    top10_df, 
    tweets_top_countries["country"] == top10_df["country"], 
    how="leftsemi"
).select(tweets_top_countries["country"], tweets_top_countries["sentiment"], tweets_top_countries["cnt"])

# Display the results sorted by count (cnt) in descending order
display(tweets_top10_countries_df.orderBy("cnt", ascending=False))

#Value of HiveQL is different here as well

# COMMAND ----------

# MAGIC %md
# MAGIC **Filter Out Rows Where Sentiment is Null**
# MAGIC
# MAGIC Filter out rows with null sentiment values and visualize the result using Bar chart for better visibility of data.

# COMMAND ----------

# Assuming 'tweets_top10_countries_df' is the DataFrame corresponding to the 'tweets_top10_countries' table

# Filter out rows where 'sentiment' is not null and order by 'cnt' in descending order
tweets_top10_countries_df_filtered = tweets_top10_countries_df \
    .filter(tweets_top10_countries_df.sentiment.isNotNull()) \
    .orderBy(tweets_top10_countries_df.cnt.desc())

# Display the results
display(tweets_top10_countries_df_filtered)

