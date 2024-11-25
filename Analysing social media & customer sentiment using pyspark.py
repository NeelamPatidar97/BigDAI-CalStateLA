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



from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Customer Sentiment Analysis") \
    .getOrCreate()

# HDFS File location
tweets_alphago_location = "hdfs:///user/npatida/tmp/tweetsbi/tweets_alphago.csv"

# Load the CSV file into a DataFrame
tweetsbi_df = spark.read.format("csv") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .option("sep", ",") \
    .load(tweets_alphago_location)

# Display the first few rows
tweetsbi_df.show(3)

# Display the schema
tweetsbi_df.printSchema()

# Filter out null, empty, and invalid values for country
Filtered_tweetsbi_df = tweetsbi_df.filter(
    (col('country').isNotNull()) &
    (col('country') != '') &
    (col('country') != 'null')
)

# Group by country and sentiment, and count occurrences
tweets_top_countries = Filtered_tweetsbi_df.groupBy('country', 'sentiment').agg(
    count('sentiment').alias('cnt')
).orderBy(col('cnt').desc())

# Display the results
tweets_top_countries.show(2)

# Group by country and sum the counts to create the top10 equivalent
top10_df = tweets_top_countries.groupBy("country") \
    .sum("cnt") \
    .withColumnRenamed("sum(cnt)", "cnt2") \
    .orderBy("cnt2", ascending=False) \
    .limit(10)

# Display the top 5 rows from the grouped DataFrame
top10_df.show(5)

# Perform a left semi join to match the SQL join operation
tweets_top10_countries_df = tweets_top_countries.join(
    top10_df, 
    tweets_top_countries["country"] == top10_df["country"], 
    how="leftsemi"
).select(tweets_top_countries["country"], tweets_top_countries["sentiment"], tweets_top_countries["cnt"])

# Display the results sorted by count (cnt) in descending order
tweets_top10_countries_df.orderBy(col("cnt").desc()).show()

# Filter out rows where 'sentiment' is not null and order by 'cnt' in descending order
tweets_top10_countries_df_filtered = tweets_top10_countries_df \
    .filter(tweets_top10_countries_df.sentiment.isNotNull()) \
    .orderBy(tweets_top10_countries_df.cnt.desc())

# Display the filtered results
tweets_top10_countries_df_filtered.show()

# Save final results to HDFS for future reference
output_path = "hdfs:///user/npatida/tmp/tweetsbi_output/"
tweets_top10_countries_df_filtered.write.csv(output_path, header=True, mode="overwrite")

print(f"Results saved to HDFS at {output_path}")
