# Databricks notebook source
# MAGIC %md
# MAGIC ### ✨ **Author Information** 
# MAGIC
# MAGIC |                    |                                                                 |
# MAGIC |--------------------|-----------------------------------------------------------------|
# MAGIC | **Author**         | Neelam Patidar                                                  |
# MAGIC | **Date**           | August 26, 2024                                                 |
# MAGIC | **Supervisor**     | Prof. Jongwook Woo                                              |
# MAGIC | **Affiliation**    | Big Data AI Center (BigDAI): High Performance Information Computing Center (HiPIC) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 📚 **Tutorial Overview**
# MAGIC
# MAGIC Social media platforms are powerful catalysts for Big Data adoption, driving insights into popular culture and public sentiment.Public APIs provided by sites like Twitter are a useful source of data for analyzing and understanding popular trends. 
# MAGIC
# MAGIC In this tutorial, guided by Prof. Woo, you will explore how to use pyspark for analyzing and visualizing Twitter data. Specifically, you'll dive into sentiment analysis of tweets about the movie **Minions**, collected between August 25 – August 28, 2015, using NiFi.
# MAGIC
# MAGIC We'll load raw JSON data, parse and transform it, and then use a sentiment dictionary to classify the sentiment of each tweet. Finally, we'll clean and filter the data to prepare it for analysis.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC In this tutorial, we'll walk through the process of performing sentiment analysis on a dataset of tweets. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Dataset Uploading to the DBFS
# MAGIC
# MAGIC Below is the link of the data that are used for this twitter sentiment analysis:
# MAGIC
# MAGIC - time_zone_map.tsv https://github.com/dalgual/aidatasci/raw/master/data/bigdata/time_zone_map.tsv
# MAGIC - dictionary.tsv https://github.com/dalgual/aidatasci/raw/master/data/bigdata/dictionary.tsv
# MAGIC - minion_tweets.tar.gz https://github.com/dalgual/aidatasci/raw/master/data/bigdata/minion_tweets.tar.gz
# MAGIC
# MAGIC Note- Extract the **tar.gz** file in your local system. You can use **7-zip file manager** application to complete the file extraction.
# MAGIC
# MAGIC Once you complete the extraction, follow the below steps to correctly upload the data to DBFS:
# MAGIC
# MAGIC **Access Databricks:**
# MAGIC - Log in to your Databricks workspace.
# MAGIC
# MAGIC **Upload Files:**
# MAGIC - Navigate to the "Catalog" tab in Databricks.
# MAGIC - Click on create table
# MAGIC - Use the "Upload File" option to upload your data files  **dictionary.tsv**, and **time_zone_map.tsv.**
# MAGIC - Once the above two file are uploaded, then upload **minion_tweets.tar.gz** file with the same upload option but make sure to mention the folder name **Tweets** to your directory.
# MAGIC
# MAGIC This process will ensure that all your extracted files of **tar.gz** will be uploaded to the Tweets folder in the databricks file system.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load and Inspect Raw Tweet Data
# MAGIC
# MAGIC First, we load the raw tweet data stored as JSON strings within CSV files. We'll then inspect the structure of the loaded DataFrame to ensure the data is correctly loaded.
# MAGIC
# MAGIC This step ensures that the data is loaded correctly and that the 'value' column, containing the JSON strings, is present.

# COMMAND ----------

# Path to the CSV files containing JSON strings
minions_json_files = "/FileStore/tables/Tweets/*.csv"  # Uses wildcard to load all CSV files in the directory

# Read the text files
raw_tweets_df = spark.read.text(minions_json_files)


# COMMAND ----------

# Display the schema to confirm the DataFrame has the correct structure
raw_tweets_df.printSchema()

# Show some initial data to make sure 'value' column exists
display(raw_tweets_df.limit(1))


# COMMAND ----------

# MAGIC %md
# MAGIC ###Parse JSON Data from Raw Tweets
# MAGIC
# MAGIC Next, we will define a schema that mirrors the structure of the JSON data stored in the CSV files and then use this schema to parse the JSON strings in the DataFrame
# MAGIC
# MAGIC The **tweets_json_schema** is created to match the expected structure of the JSON data. The **from_json** function is then used to parse the JSON strings into a structured format.

# COMMAND ----------

from pyspark.sql.functions import from_json, col, get_json_object, substring, when

# Define a schema for the JSON data based on your Hive table structure
from pyspark.sql.types import StringType, StructType, StructField, LongType

tweets_json_schema = StructType([
    StructField("id_str", StringType()),
    StructField("created_at", StringType()),
    StructField("text", StringType()),
    StructField("user", StructType([
        StructField("time_zone", StringType())
    ]))
])

# Parse the JSON data
parsed_tweets_df = raw_tweets_df.withColumn("json_data", from_json(col("value"), tweets_json_schema))


# COMMAND ----------

# MAGIC %md
# MAGIC ###Select and Transform Tweet Data
# MAGIC
# MAGIC Here, we will select specific fields from the parsed JSON data and apply transformations to format date and time fields.
# MAGIC
# MAGIC This step transforms the raw date into a more manageable format by extracting components such as year, month, and day, making it easier to perform time-based analysis .
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import expr

# Select and transform data
tweets_text_df = parsed_tweets_df.select(
    col("json_data.id_str").cast(LongType()).alias("id"),
    col("json_data.created_at").alias("created_at"),
    expr("concat(substring(json_data.created_at, 1, 10), ' ', substring(json_data.created_at, 27, 4))").alias("created_at_date"),
    substring(col("json_data.created_at"), 27, 4).alias("created_at_year"),
    expr("""
        case substring(json_data.created_at, 5, 3)
            when 'Jan' then '01'
            when 'Feb' then '02'
            when 'Mar' then '03'
            when 'Apr' then '04'
            when 'May' then '05'
            when 'Jun' then '06'
            when 'Jul' then '07'
            when 'Aug' then '08'
            when 'Sep' then '09'
            when 'Oct' then '10'
            when 'Nov' then '11'
            when 'Dec' then '12'
        end
    """).alias("created_at_month"),
    substring(col("json_data.created_at"), 9, 2).alias("created_at_day"),
    substring(col("json_data.created_at"), 12, 8).alias("created_at_time"),
    col("json_data.text").alias("text"),
    col("json_data.user.time_zone").alias("time_zone")
)

# Show the resulting DataFrame
display(tweets_text_df.limit(10))


# COMMAND ----------

# MAGIC %md
# MAGIC ###Load and Transform Supporting Data (Time Zone and Dictionary Data)
# MAGIC
# MAGIC In this step, we load additional data needed for the analysis, including a time zone mapping file and a sentiment dictionary. We will also rename the columns in the dictionary for better readability.
# MAGIC
# MAGIC The time zone and dictionary data are loaded and prepared for use in joining and sentiment analysis tasks.

# COMMAND ----------

# File location and type
time_zone_map_file = "/FileStore/tables/time_zone_map.tsv"
file_type = "csv"  

# CSV options
infer_schema = "true" 
first_row_is_header = "true"
delimiter = "\t"

# Load TSV data
time_zone_df= spark.read.format(file_type) \
    .option("inferSchema", infer_schema) \
    .option("header", first_row_is_header) \
    .option("sep", delimiter) \
    .load(time_zone_map_file)

display(time_zone_df.limit(10))

# COMMAND ----------

from pyspark.sql.functions import col

# File location and type
dictionary_file = "/FileStore/tables/dictionary.tsv"
file_type = "csv"  # Using csv since Spark CSV can handle TSV with delimiter set to '\t'

# CSV options
infer_schema = "true" 
first_row_is_header = "false"
delimiter = "\t"

# Load TSV data
dictionary_df = spark.read.format(file_type) \
    .option("inferSchema", infer_schema) \
    .option("header", first_row_is_header) \
    .option("sep", delimiter) \
    .load(dictionary_file)

# Rename columns in the dictionary DataFrame for better readability
dictionary_df = dictionary_df.select(
    col("_c0").alias("strength"),
    col("_c1").alias("length"), # This seems unused based on the screenshot, replace as needed
    col("_c2").alias("word"),
    col("_c3").alias("part_of_speech"),
    col("_c4").alias("stemmed"),  # Indicates if the word is stemmed
    col("_c5").alias("polarity")  # Sentiment polarity of the word
)

# Display the DataFrame to verify changes
display(dictionary_df.limit(10))


# COMMAND ----------

# Print the schema to see all column names
tweets_text_df.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a Simple Tweets DataFrame with Timestamp
# MAGIC
# MAGIC Here, we will create a simplified version of the tweet data by selecting key fields and transforming the date and time components into a single timestamp.
# MAGIC
# MAGIC This DataFrame simplifies the tweet data by combining the date and time into a single **timestamp** field, making it easier to work with for time-based analysis.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import concat, col, lit, substring

# Print intermediate value to check concatenation result
tweets_simple_df = tweets_text_df.select(
    col("id").alias("tweet__id"),
    concat(col("created_at_year"), lit(" "), col("created_at_month"), lit(" "), substring(col("created_at"), 9, 2), lit(" "), col("created_at_time")).alias("timestamp"),
    col("text").alias("msg"),
    col("time_zone")
)

display(tweets_simple_df.limit(10))


# COMMAND ----------

# MAGIC %md
# MAGIC ###Join Tweets Data with Time Zone Mapping
# MAGIC
# MAGIC In this step, we will enrich the tweet data by joining it with the time zone mapping data to associate each tweet with a country.
# MAGIC
# MAGIC The **tweets_clean_df** DataFrame now includes a country for each tweet, based on the time zone, enabling geographical analysis.

# COMMAND ----------

# Ensure the correct column names are used in the join and selection
tweets_clean_df = tweets_simple_df.alias("t").join(
    time_zone_df.alias("m"),
    col("t.time_zone") == col("m.time_zone"),
    "left_outer"
).select(
    col("t.tweet__id").alias("tweet_id"),
    col("t.timestamp").alias("ts"),  # Using 'timestamp' as per your schema
    col("t.msg"),
    col("m.country")
)

# Display the DataFrame to verify the final result
display(tweets_clean_df.limit(10))


# COMMAND ----------

# MAGIC %md
# MAGIC ###Explode Tweet Text into Individual Words
# MAGIC
# MAGIC Now, we will process the tweet text by converting it to lowercase, splitting it into individual words, and then exploding these words into separate rows.
# MAGIC
# MAGIC This step prepares the text data for sentiment analysis by breaking down each tweet into its individual words.

# COMMAND ----------

from pyspark.sql.functions import explode, split, lower

# Explode the text into words
l1_df = tweets_text_df.select(
    col("id"),
    explode(split(lower(col("text")), "\\s+")).alias("words")
)

# Display the resulting DataFrame to verify the output
display(l1_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Further Explode Words into Individual Word Entries
# MAGIC
# MAGIC Further, we continue processing the tweet text by exploding the words into individual word entries, setting the stage for sentiment analysis.
# MAGIC
# MAGIC The **l2_df** DataFrame now contains one row per word per tweet, allowing for detailed word-level sentiment analysis.

# COMMAND ----------

from pyspark.sql.functions import explode, split, col

# Step 1: Create l1_df by splitting the text into words
l1_df = tweets_text_df.select(
    col("id"),
    split(col("text"), "\\s+").alias("words")  # Split the text into an array of words
)

# Step 2: Create l2_df by exploding the array into individual words
l2_df = l1_df.select(
    col("id"),
    explode(col("words")).alias("word")  # Explode the words array into individual words
)

# Display the resulting DataFrame to verify the output
display(l2_df.limit)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Compute Sentiment Polarity for Each Word
# MAGIC
# MAGIC Now, we will compute the sentiment polarity of each word in the tweets by joining the exploded words with a sentiment dictionary.
# MAGIC
# MAGIC We join the words with the dictionary to determine the sentiment of each word, allowing us to later aggregate these sentiments to determine the overall sentiment of each tweet.

# COMMAND ----------

from pyspark.sql.functions import when

# Join l2 with the dictionary to compute sentiment polarity
l3_df = l2_df.alias("l2").join(
    dictionary_df.alias("d"),
    col("l2.word") == col("d.word"),
    "left_outer"
).select(
    col("l2.id").alias("tweet_id"),
    col("l2.word"),
    when(col("d.polarity") == "negative", -1)
    .when(col("d.polarity") == "positive", 1)
    .otherwise(0).alias("polarity")
)

# Display the resulting DataFrame to verify the output
display(l3_df.limit(3))


# COMMAND ----------

# MAGIC %md
# MAGIC ###Calculate Overall Sentiment for Each Tweet
# MAGIC
# MAGIC Finally, we calculate the overall sentiment for each tweet by aggregating the sentiment polarities of individual words.

# COMMAND ----------

from pyspark.sql.functions import sum, when, col

# Create the tweets_sentiment DataFrame by calculating the sentiment based on polarity
tweets_sentiment_df = l3_df.groupBy("tweet_id").agg(
    when(sum("polarity") > 0, "positive")
    .when(sum("polarity") < 0, "negative")
    .otherwise("neutral").alias("sentiment")
)

# Display the DataFrame to verify
display(tweets_sentiment_df.limit(3))


# COMMAND ----------

# MAGIC %md
# MAGIC ###Combine Tweet Data with Sentiment Scores
# MAGIC
# MAGIC Now, we combine the clean tweet data with the calculated sentiment scores to create a final DataFrame that includes both the original tweet information and the sentiment analysis result.
# MAGIC
# MAGIC This step integrates the results of the sentiment analysis into the main tweet dataset, providing a comprehensive view of each tweet along with its sentiment classification.

# COMMAND ----------

# Join tweets_clean_df with tweets_sentiment_df
tweetsbi_df = tweets_clean_df.alias("t").join(
    tweets_sentiment_df.alias("s"),
    col("t.tweet_id") == col("s.tweet_id"),
    "left_outer"
).select(
    col("t.*"),
    when(col("s.sentiment") == "positive", 2)
    .when(col("s.sentiment") == "neutral", 1)
    .when(col("s.sentiment") == "negative", 0)
    .alias("sentiment")
)

# Display the DataFrame to verify
display(tweetsbi_df.limit(3))


# COMMAND ----------

# MAGIC %md
# MAGIC ###Count the Number of Records in the Final DataFrame
# MAGIC
# MAGIC Now, we count the number of records in the **tweetsbi_df** DataFrame to verify the total number of tweets that have been processed and included in the final dataset which contains both the tweet information and the sentiment scores.

# COMMAND ----------

# Count the number of records in tweetsbi_df
record_count = tweetsbi_df.count()

# Display the result
print(f"Total records in tweetsbi: {record_count}")


# COMMAND ----------

# MAGIC %md
# MAGIC ###Filter and Clean the Final Sentiment Data
# MAGIC
# MAGIC In this step, we finalize the sentiment analysis process by filtering the tweet data to include only valid records and cleaning the message text.
# MAGIC
# MAGIC This step ensures that the sentiment data is clean, consistent, and ready for further analysis or storage. By removing irrelevant characters from the text and filtering out records with invalid sentiment values, the data is now in a state that can be reliably used for reporting or further data processing.

# COMMAND ----------

from pyspark.sql.functions import regexp_replace, col

# Filter the DataFrame and clean the message column
tweetsent_df = tweetsbi_df.filter(
    (col("tweet_id").isNotNull()) &
    (col("sentiment").isin(0, 1, 2))
).select(
    col("tweet_id"),
    col("ts"),
    regexp_replace(col("msg"), '[^a-zA-Z0-9]+', ' ').alias("message"),
    col("country"),
    col("sentiment")
)

# Display the resulting DataFrame to verify the output
display(tweetsent_df.limit(10))

