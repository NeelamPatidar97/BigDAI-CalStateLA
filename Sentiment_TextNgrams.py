from pyspark.sql import SparkSession
import re
from pyspark.ml.feature import Tokenizer, NGram
from pyspark.sql.functions import col, lower, explode, size, concat_ws
from pyspark.sql import functions as F
from pyspark.sql.functions import col, avg, count, desc


# Create or retrieve a Spark session
spark = SparkSession.builder.appName("Customer Sentiment Analysis TextNgrams").getOrCreate()


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

ratings_2012_df.show()



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

ratings_2013_df.show()



# Count the number of records in ratings_2012_df
ratings_2012_count = ratings_2012_df.count()
print(f"Number of records in 2012 ratings: {ratings_2012_count}")

# Count the number of records in ratings_2013_df
ratings_2013_count = ratings_2013_df.count()
print(f"Number of records in 2013 ratings: {ratings_2013_count}")



# Combine the 2012 and 2013 ratings DataFrames
ratings_df = ratings_2012_df.union(ratings_2013_df)
ratings_df.show()




# Count the total number of records after merging
total_count = ratings_df.count()
print(f"Total number of records: {total_count}")




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

products_df.show(2)



# Show the schema of the products DataFrame
products_df.printSchema()






# Find the product with the highest average rating among those with at least 50 ratings


# Calculate average ratings and filter products with at least 50 ratings
avg_ratings_df = ratings_df.groupBy("prod_id") \
    .agg(avg("rating").alias("avg_rating"), count("*").alias("num")) \
    .filter(col("num") >= 50) \
    .orderBy(desc("avg_rating"))

# Show the top product
top_product = avg_ratings_df.limit(1)
top_product.show()




# Find the product with the lowest average rating among those with at least 50 ratings
lowest_avg_ratings_df = avg_ratings_df.orderBy(col("avg_rating"))
bottom_product = lowest_avg_ratings_df.limit(1)
bottom_product.show()






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
bigram_counts_df.show(5)



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
trigram_counts_df.show(5)



# Filter the DataFrame for the specified product ID and messages containing the phrase "ten times more"
filtered_comments_df = ratings_df.filter(
    (col("prod_id") == 1274673) & 
    (col("message").like('%ten times more%'))
)

# Display the top 3 comments that match the condition
filtered_comments_df.select("message").show(3)





# Filter the DataFrame for the specified product ID and comments containing the word "red"
distinct_comments_df = ratings_df.filter(
    (col("prod_id") == 1274673) & 
    (col("message").like('%red%'))
).select("message").distinct()

# Display the distinct comments
distinct_comments_df.show()






# Filter the products DataFrame to display the record for product ID 1274673
product_record_df = products_df.filter(col("prod_id") == 1274673)

# Display the product record
product_record_df.show()





# Filter the products DataFrame for the specific name and brand conditions
filtered_products_df = products_df.filter(
    (col("name").like('%16 GB USB Flash Drive%')) & 
    (col("brand") == 'Orion')
)

# Display the filtered results
filtered_products_df.show()






# Step 1: Convert the message column to lowercase and tokenize it into individual words
tokenizer = Tokenizer(inputCol="message", outputCol="words")
tokenized_df = tokenizer.transform(ratings_df.withColumn("message", lower(col("message"))))

# Display the tokenized words (1-grams)
tokenized_df.select("message", "words").show()




# Explode the words to create individual rows for each 1-gram
onegram_df = tokenized_df.select(explode(col("words")).alias("onegram"))

# Display the exploded 1-gram results
onegram_df.show()




# Generate 2-grams (bigrams) from the tokenized words
ngram_2 = NGram(n=2, inputCol="words", outputCol="bigrams")
bigrams_df = ngram_2.transform(tokenized_df)

# Explode the bigrams to display each bigram in a separate row
exploded_bigrams_df = bigrams_df.select(explode(col("bigrams")).alias("bigram"))

# Display the exploded bigram results
exploded_bigrams_df.show()



# Generate 4-grams (n=4) from the tokenized words
ngram_4 = NGram(n=4, inputCol="words", outputCol="fourgrams")
fourgrams_df = ngram_4.transform(tokenized_df)

# Explode the 4-grams to create individual rows for each 4-gram
exploded_fourgrams_df = fourgrams_df.select(explode(col("fourgrams")).alias("fourgram"))

# Filter for 4-grams that specifically start with "red one"
context_snippets_df = exploded_fourgrams_df.filter(col("fourgram").rlike(r'^red one'))

# Display the filtered 4-grams that match the context
context_snippets_df.show(10)







def context_ngram(tokenized_df, context_prefix, input_col="words"):
    
    # Validate 'context_prefix' parameter
    if not context_prefix or context_prefix.strip() == "":
        raise ValueError("Context prefix cannot be empty. Please provide a valid string.")
    
    # Split context prefix into words to determine n-gram size
    context_words = context_prefix.split()
    n = len(context_words)
    
    # Replace 'null' placeholders with '.*' to create a regex pattern for context filtering
    regex_pattern = "^" + ' '.join([re.escape(word) if word != "null" else ".*" for word in context_words])
    
    # Generate n-grams based on the length of the context prefix
    ngram = NGram(n=n, inputCol=input_col, outputCol="ngrams")
    ngram_df = ngram.transform(tokenized_df)
    
    # Explode n-grams to create individual rows for each n-gram
    exploded_ngram_df = ngram_df.select(explode(col("ngrams")).alias("ngram"))
    
    # Filter n-grams based on the context regex pattern
    filtered_ngram_df = exploded_ngram_df.filter(col("ngram").rlike(regex_pattern))
    
    # Count frequency of each n-gram and sort by frequency in descending order
    result_df = (
        filtered_ngram_df
        .groupBy("ngram")
        .count()
        .withColumnRenamed("count", "estfrequency")
        .orderBy(col("estfrequency").desc())
    )
    
    # Display results in the desired format
    # Display the top 10 results, regardless of available matches.
    result_df.show(10)

    return result_df

# Function Call
# This call generates context-based n-grams with a flexible context prefix
context_ngram(tokenized_df, context_prefix="quality null null", input_col="words")

