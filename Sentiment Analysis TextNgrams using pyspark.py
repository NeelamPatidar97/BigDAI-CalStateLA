# Databricks notebook source
# MAGIC %md
# MAGIC ### ✨ **Author Information** 
# MAGIC
# MAGIC |                    |                                                                 |
# MAGIC |--------------------|-----------------------------------------------------------------|
# MAGIC | **Author**         | Neelam Patidar                                                  |
# MAGIC | **Date**           | September 7, 2024                                                 |
# MAGIC | **Supervisor**     | Prof. Jongwook Woo                                              |
# MAGIC | **Affiliation**    | Big Data AI Center (BigDAI): High Performance Information Computing Center (HiPIC) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 📚 **Tutorial Overview**
# MAGIC The business data set and its associated customer review data are among the most common applications of Big Data. In this tutorial, you will learn how to use PySpark in Databricks to analyze sentiment and gain insights from user data related to product ratings and reviews from the DualCore example data set. You will perform data processing, text analysis, and n-gram generation to explore how customers evaluate products, revealing patterns and sentiment trends

# COMMAND ----------

# MAGIC %md
# MAGIC ###Introduction
# MAGIC
# MAGIC In this tutorial, we will leverage PySpark, an essential Big Data processing tool, to analyze business data files and gain insights into how customers perceive and evaluate products from DualCore. By performing text processing and sentiment analysis on customer comments and ratings, you can uncover valuable insights about product performance and customer satisfaction. The tutorial will guide you through loading data, cleaning and transforming it, and using advanced text processing features to analyze customer feedback and product ratings.
# MAGIC
# MAGIC In this tutorial, you'll learn how to:
# MAGIC - Download and upload multiple CSV and TSV files
# MAGIC - Prepare and clean data
# MAGIC - Combine data from different sources
# MAGIC - Create and manipulate DataFrames using PySpark
# MAGIC - Generate n-grams (bigrams and trigrams) to analyze customer feedback
# MAGIC - Perform sentiment analysis on customer comments and product ratings
# MAGIC - Filter and explore specific phrases and terms
# MAGIC - Visualize and interpret analysis results

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Dataset Uploading to the DBFS
# MAGIC
# MAGIC Below is the link of the data that are used for this twitter sentiment analysis:
# MAGIC
# MAGIC - ratings_2012.txt https://github.com/dalgual/aidatasci/raw/master/data/bigdata/ratings_2012.txt
# MAGIC - ratings_2013.txt https://github.com/dalgual/aidatasci/raw/master/data/bigdata/ratings_2013.txt
# MAGIC - products.tsv https://github.com/dalgual/aidatasci/raw/master/data/bigdata/products.tsv
# MAGIC
# MAGIC Follow the below steps to correctly upload the data to DBFS:
# MAGIC
# MAGIC **Access Databricks:**
# MAGIC - Log in to your Databricks workspace.
# MAGIC
# MAGIC **Upload Files:**
# MAGIC - Navigate to the "Catalog" tab in Databricks.
# MAGIC - Click on create table
# MAGIC - Use the "Upload File" option to upload your data files
# MAGIC - Make sure to mention the folder name **"ratings"**, **"dualcore"** & **"products"** to your directory for all 3 files respectively.
# MAGIC
# MAGIC This process will ensure that all your files will be uploaded to the respective folder in the databricks file system.

# COMMAND ----------

# MAGIC %md
# MAGIC **Load and Prepare Ratings Data for 2012**
# MAGIC
# MAGIC First, load the ratings data for the year 2012. This step involves specifying the file location and format, then reading the data into a DataFrame with appropriate schema inference and delimiter options. Finally, rename the columns to make the data more readable.

# COMMAND ----------

from pyspark.sql.functions import col

# File location and type
ratings_file_location = "/FileStore/tables/ratings/ratings_2012.txt"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "false" 
delimiter = "\t" 

# The applied options are for CSV files formatted as TSV.
ratings_2012_df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(ratings_file_location)
  
# Rename columns in the ratings_2012_df DataFrame for better readability
ratings_2012_df = ratings_2012_df.select(
    col("_c0").alias("posted"),
    col("_c1").alias("cust_id"), 
    col("_c2").alias("prod_id"),
    col("_c3").alias("rating"),
    col("_c4").alias("message")
)

display(ratings_2012_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Load and Prepare Ratings Data for 2013**
# MAGIC
# MAGIC Repeat the process for the 2013 ratings data, ensuring that it is formatted and renamed consistently with the 2012 data.

# COMMAND ----------

# File location and type
dualcore_file_location = "/FileStore/tables/dualcore/ratings_2013.txt"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "false"  # Adjust this based on whether your TSV file includes column names
delimiter = "\t"  

# The applied options are for CSV files formatted as TSV.
ratings_2013_df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(dualcore_file_location)

# Rename columns in the ratings_2013_df DataFrame for better readability
ratings_2013_df = ratings_2013_df.select(
    col("_c0").alias("posted"),
    col("_c1").alias("cust_id"), 
    col("_c2").alias("prod_id"),
    col("_c3").alias("rating"),
    col("_c4").alias("message")
)

display(ratings_2013_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Count Records in Each DataFrame**
# MAGIC
# MAGIC Count the number of records in each of the 2012 and 2013 DataFrames to ensure data integrity.
# MAGIC

# COMMAND ----------

# Count the number of records in ratings_2012_df
ratings_2012_count = ratings_2012_df.count()
print(f"Number of records in 2012 ratings: {ratings_2012_count}")

# Count the number of records in ratings_2013_df
ratings_2013_count = ratings_2013_df.count()
print(f"Number of records in 2013 ratings: {ratings_2013_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Combine the 2012 and 2013 Ratings DataFrames into one for comprehensive analysis**

# COMMAND ----------

# Combine the 2012 and 2013 ratings DataFrames
ratings_df = ratings_2012_df.union(ratings_2013_df)
display(ratings_df)


# COMMAND ----------

# MAGIC %md
# MAGIC Check the total number of records to ensure that the merge was successful.

# COMMAND ----------

# Count the total number of records after merging
total_count = ratings_df.count()
print(f"Total number of records: {total_count}")


# COMMAND ----------

# MAGIC %md
# MAGIC **Load and Prepare Product Data**
# MAGIC
# MAGIC Load the product data, which contains product details like brand, name, and price, and rename the columns.

# COMMAND ----------

# File location and type
products_file_location = "/FileStore/tables/products/products.tsv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "false"  # Adjust this based on whether your TSV file includes column names
delimiter = "\t" 

# The applied options are for CSV files formatted as TSV.
products_df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(products_file_location)

# Rename columns in the products DataFrame for better readability
products_df = products_df.select(
    col("_c0").alias("prod_id"),
    col("_c1").alias("brand"), 
    col("_c2").alias("name"),
    col("_c3").alias("price"),
    col("_c4").alias("cost"),  
    col("_c5").alias("shipping_wt")
)

display(products_df.limit(2))

# COMMAND ----------

# Show the schema of the products DataFrame
products_df.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC **Analyze Product Ratings**
# MAGIC
# MAGIC We want to find the product that customers like most, but must guard against being misled by products that have few ratings assigned. Run the following query to find the product with the highest average using DESC among all those with at least 50 ratings:

# COMMAND ----------

# Find the product with the highest average rating among those with at least 50 ratings
from pyspark.sql.functions import col, avg, count, desc

# Calculate average ratings and filter products with at least 50 ratings
avg_ratings_df = ratings_df.groupBy("prod_id") \
    .agg(avg("rating").alias("avg_rating"), count("*").alias("num")) \
    .filter(col("num") >= 50) \
    .orderBy(desc("avg_rating"))

# Show the top product
top_product = avg_ratings_df.limit(1)
display(top_product)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Now, below is the query to find the product with the lowest average among products with at least 50 ratings (num >= 50). 

# COMMAND ----------

# Find the product with the lowest average rating among those with at least 50 ratings
lowest_avg_ratings_df = avg_ratings_df.orderBy(col("avg_rating"))
bottom_product = lowest_avg_ratings_df.limit(1)
display(bottom_product)


# COMMAND ----------

# MAGIC %md
# MAGIC We observed earlier that customers are very dissatisfied with one of the products that Dualcore sells. Although numeric ratings can help identify which product that is, they don’t tell Dualcore why customers don’t like the product. We could simply read through all the comments associated with that product to learn this information, but that approach doesn’t scale. Next, you will use Hive’s text processing support to analyze the comments.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The following query normalizes all comments on that product to lowercase, breaks them into individual words using the *SENTENCES* function, and passes those to the *NGRAMS* function to find the five most common bigrams (two-‐ word combinations).

# COMMAND ----------

from pyspark.ml.feature import Tokenizer, NGram
from pyspark.sql.functions import col, lower, explode, size, concat_ws
from pyspark.sql import functions as F

# Filter the ratings DataFrame for the specified product ID
product_ratings_df = ratings_df.filter(col("prod_id") == 1274673)

# Convert messages to lowercase
product_ratings_df = product_ratings_df.withColumn("message_lower", lower(col("message")))

# Tokenize the messages into words
tokenizer = Tokenizer(inputCol="message_lower", outputCol="words")
tokenized_df = tokenizer.transform(product_ratings_df)

# Generate bigrams (n=2) from the tokenized words
ngram = NGram(n=2, inputCol="words", outputCol="bigrams")
bigrams_df = ngram.transform(tokenized_df)

# Explode the bigrams to get each bigram in a separate row and count their frequencies
exploded_bigrams_df = bigrams_df.select(explode(col("bigrams")).alias("bigram"))

# Count the frequency of each bigram
bigram_counts_df = exploded_bigrams_df.groupBy("bigram").count() \
    .withColumnRenamed("count", "estfrequency") \
    .orderBy(F.desc("estfrequency"))

# Display the result in the required format, showing the top 5 bigrams
display(bigram_counts_df.limit(5))


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC Most of these words are too common to provide much insight, though the word **“expensive”** does stand out in the list. Modify the previous query to find the five most common **trigrams** (three-‐ word combinations)

# COMMAND ----------

from pyspark.ml.feature import Tokenizer, NGram
from pyspark.sql.functions import col, lower, explode
from pyspark.sql import functions as F

# Filter the ratings DataFrame for the specified product ID
product_ratings_df = ratings_df.filter(col("prod_id") == 1274673)

# Convert messages to lowercase
product_ratings_df = product_ratings_df.withColumn("message_lower", lower(col("message")))

# Tokenize the messages into words
tokenizer = Tokenizer(inputCol="message_lower", outputCol="words")
tokenized_df = tokenizer.transform(product_ratings_df)

# Generate trigrams (n=3) from the tokenized words
ngram = NGram(n=3, inputCol="words", outputCol="trigrams")
trigrams_df = ngram.transform(tokenized_df)

# Explode the trigrams to get each trigram in a separate row
exploded_trigrams_df = trigrams_df.select(explode(col("trigrams")).alias("trigram"))

# Count the frequency of each trigram
trigram_counts_df = exploded_trigrams_df.groupBy("trigram").count() \
    .withColumnRenamed("count", "estfrequency") \
    .orderBy(F.desc("estfrequency"))

# Display the top 5 trigrams with their estimated frequency
display(trigram_counts_df.limit(5))


# COMMAND ----------

# MAGIC %md
# MAGIC Among the patterns you see in the result is the phrase “ten times more.” This might be related to the complaints that the product is too expensive. Now that you’ve identified a specific phrase, look at a few comments that contain it by running the below code:

# COMMAND ----------

from pyspark.sql.functions import col

# Filter the DataFrame for the specified product ID and messages containing the phrase "ten times more"
filtered_comments_df = ratings_df.filter(
    (col("prod_id") == 1274673) & 
    (col("message").like('%ten times more%'))
)

# Display the top 3 comments that match the condition
display(filtered_comments_df.select("message").limit(3))


# COMMAND ----------

# MAGIC %md
# MAGIC We can infer that customers are complaining about the price of this item, but the comment alone doesn’t provide enough detail. One of the words (“red”) in that comment was also found in the list of trigrams from the earlier query. Run below query that will find all distinct comments containing the word “red” that are associated with product ID 1274673

# COMMAND ----------

from pyspark.sql.functions import col

# Filter the DataFrame for the specified product ID and comments containing the word "red"
distinct_comments_df = ratings_df.filter(
    (col("prod_id") == 1274673) & 
    (col("message").like('%red%'))
).select("message").distinct()

# Display the distinct comments
display(distinct_comments_df)


# COMMAND ----------

# MAGIC %md
# MAGIC The previous step should have displayed two comments:
# MAGIC
# MAGIC • “What is so special about red?”
# MAGIC
# MAGIC • “Why does the red one cost ten times more than the others?”
# MAGIC
# MAGIC The second comment implies that this product is overpriced relative to similar products. Following query will display the record for product ID 1274673 

# COMMAND ----------

from pyspark.sql.functions import col

# Filter the products DataFrame to display the record for product ID 1274673
product_record_df = products_df.filter(col("prod_id") == 1274673)

# Display the product record
display(product_record_df)


# COMMAND ----------

# MAGIC %md
# MAGIC Your query should have shown that the product was a “16GB USB Flash Drive (Red)” from the “Orion” brand. Next, run this query to identify similar products:

# COMMAND ----------

from pyspark.sql.functions import col

# Filter the products DataFrame for the specific name and brand conditions
filtered_products_df = products_df.filter(
    (col("name").like('%16 GB USB Flash Drive%')) & 
    (col("brand") == 'Orion')
)

# Display the filtered results
display(filtered_products_df)


# COMMAND ----------

# MAGIC %md
# MAGIC The query results show that there are three almost identical products, but the product with the negative reviews (the red one) costs about ten times as much as the others, just as some of the comments said.
# MAGIC Based on the cost and price columns, it appears that doing text processing on the product ratings has helped Dualcore uncover a pricing error.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **Please run the following code to find out any words about red as well:**
# MAGIC
# MAGIC Step 1: Extract 1-Grams (Words) and Display
# MAGIC
# MAGIC This step involves tokenizing the text into individual words and exploding them to create 1-grams.

# COMMAND ----------

from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import col, lower, explode

# Step 1: Convert the message column to lowercase and tokenize it into individual words
tokenizer = Tokenizer(inputCol="message", outputCol="words")
tokenized_df = tokenizer.transform(ratings_df.withColumn("message", lower(col("message"))))

# Display the tokenized words (1-grams)
display(tokenized_df.select("message", "words"))


# COMMAND ----------

# MAGIC %md
# MAGIC Step 2: Explode the 1-Grams to Display Individual Words
# MAGIC
# MAGIC This step will explode the tokenized words to display each word (1-gram) in a separate row, similar to Hive’s EXPLODE function.

# COMMAND ----------

# Explode the words to create individual rows for each 1-gram
onegram_df = tokenized_df.select(explode(col("words")).alias("onegram"))

# Display the exploded 1-gram results
display(onegram_df)


# COMMAND ----------

# MAGIC %md
# MAGIC Step 3: Extract 2-Grams (Bigrams) and Display
# MAGIC
# MAGIC To extract 2-grams, you will generate bigrams from the tokenized words and then explode them.

# COMMAND ----------

from pyspark.ml.feature import NGram

# Generate 2-grams (bigrams) from the tokenized words
ngram_2 = NGram(n=2, inputCol="words", outputCol="bigrams")
bigrams_df = ngram_2.transform(tokenized_df)

# Explode the bigrams to display each bigram in a separate row
exploded_bigrams_df = bigrams_df.select(explode(col("bigrams")).alias("bigram"))

# Display the exploded bigram results
display(exploded_bigrams_df)


# COMMAND ----------

# MAGIC %md
# MAGIC Step 4: Extract 4-Grams Starting with “red one”
# MAGIC
# MAGIC To find 4-grams that specifically start with "red one," generate 4-grams and filter for context.

# COMMAND ----------

# Generate 4-grams (n=4) from the tokenized words
ngram_4 = NGram(n=4, inputCol="words", outputCol="fourgrams")
fourgrams_df = ngram_4.transform(tokenized_df)

# Explode the 4-grams to create individual rows for each 4-gram
exploded_fourgrams_df = fourgrams_df.select(explode(col("fourgrams")).alias("fourgram"))

# Filter for 4-grams that specifically start with "red one"
context_snippets_df = exploded_fourgrams_df.filter(col("fourgram").rlike(r'^red one'))

# Display the filtered 4-grams that match the context
display(context_snippets_df.limit(10))


# rlike(r'^red one$')
# ^: Anchors the match at the start of the string.
# $: Anchors the match at the end of the string, ensuring no characters follow.
# r: Specifies the string as a raw string, preventing Python from interpreting backslashes and other metacharacters as escape sequences.

#Regex Pattern in rlike(): The code uses rlike(r'^red one') to filter rows in the DataFrame where the fourgram column values start with the phrase "red one". The use of a raw string ensures that the ^ symbol is treated as the start-of-string anchor in the regex, not as any special character in a regular Python string.

# COMMAND ----------

# MAGIC %md
# MAGIC Created a function context_ngram

# COMMAND ----------

# Define the function context_ngram to encapsulate the process
def context_ngram(tokenized_df, n=4, context_prefix="red one"):
    # Split the context_prefix into words and check if their count is less than n
    context_words = context_prefix.split()
    if len(context_words) > n:
        raise ValueError(f"The context prefix '{context_prefix}' contains more words than the specified n-gram length {n}.")

    # Generate n-grams
    ngram = NGram(n=n, inputCol="words", outputCol="ngrams")
    ngram_df = ngram.transform(tokenized_df)
    
    # Explode the n-grams to create individual rows
    exploded_ngram_df = ngram_df.select(explode(col("ngrams")).alias("ngram"))
    
    # Filter for n-grams that start with the specific context prefix
    context_snippets_df = exploded_ngram_df.filter(col("ngram").rlike(r'^red one'))
    
    # Display the results
    context_snippets_df.show(10)
    
    return context_snippets_df

# Usage of the function with conditional checking
context_words = "red one".split()
n = 4
if len(context_words) <= n:
    context_ngram(tokenized_df, n=n, context_prefix="red one")
else:
    print(f"Error: The context prefix 'red one' contains more words than the allowed n-gram length {n}.")

#context_ngram: The function name.
#tokenized_df: The DataFrame containing tokenized words (each row should have a column of tokens).
#n: The number of tokens in each n-gram (default is 4 for 4-grams).
#context_prefix: The prefix used to filter n-grams ("red one").
#return: The function returns a DataFrame with filtered n-grams.

