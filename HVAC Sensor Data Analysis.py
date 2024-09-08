# Databricks notebook source
# MAGIC %md
# MAGIC ### Introduction
# MAGIC
# MAGIC Many personal and commercial devices now contain sensors that collect information from the physical world, an area often referred to as the Internet of Things (IoT). For example:
# MAGIC - Most phones have a GPS.
# MAGIC - Fitness devices track how many steps you've taken.
# MAGIC - Thermostats can monitor the temperature of a building.
# MAGIC
# MAGIC In this tutorial, you'll learn how Databricks, utilizing its DBFS and PySpark, can be used to process historical data produced by heating, ventilation, and air conditioning (HVAC) systems to identify systems that are not able to reliably maintain a set temperature, a popular IoT data analysis task. You will learn how to:
# MAGIC - Refine and enrich temperature data from buildings in several countries.
# MAGIC - Analyze the data to determine which buildings have problems maintaining comfortable temperatures (actual recorded temperature vs. the temperature the thermostat was set to).
# MAGIC - Infer the reliability of HVAC systems used in the buildings.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Objective
# MAGIC
# MAGIC In this lab, you will analyze and visualize **sensor data**. Thus, you will:
# MAGIC
# MAGIC - Then, learn how to upload sensor data to the DBFS (Databricks File System).
# MAGIC - Figure out how to manipulate and analyze sensor data in DBFS using PySpark.
# MAGIC - Practice how to visualize the result in Databricks.
# MAGIC - This notebook demonstrates how to load, process, and display data using PySpark in Databricks. 
# MAGIC - It covers reading CSV files, manipulating DataFrames, and performing joins.
# MAGIC
# MAGIC This tutorial used [SensorFiles.zip](https://github.com/dalgual/aidatasci/raw/master/data/bigdata/SensorFiles.zip) dataset from Github.
# MAGIC
# MAGIC - first, You need to download the above zip file to your local system and extract files from it.
# MAGIC - Then, upload the two excel files **(HVAC.csv & building.csv)** to DBFS(Databricks File Systems).

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Importing & Displaying Data**
# MAGIC
# MAGIC - Ensure you have uploaded your CSV files to the Databricks FileStore. This example uses two files: `HVAC.csv` &`building.csv`
# MAGIC - Specify the locations and types of the files you will be working with. This step ensures that your data can be accurately located and read by Spark
# MAGIC - Set 'infer_schema' & 'first_row_is_header' to true. These options help Spark correctly interpret the structure of the files.
# MAGIC - Read the CSV files into DataFrames with predefined schemas and options to correctly interpret the headers and delimiters.
# MAGIC - Then, display the dataframes.

# COMMAND ----------

# File locations and type
file_location1 = "/FileStore/tables/HVAC.csv"
file_location2 = "/FileStore/tables/building.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# Read the first CSV file into a DataFrame
df1 = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location1)

# Read the second CSV file into a DataFrame
df2 = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location2)

# Display the DataFrames
display(df1)
display(df2)


# COMMAND ----------

# MAGIC %md
# MAGIC Create temporary views of your DataFrames.This allows you to perform SQL operations directly on the data.

# COMMAND ----------

# Create a view or table for each DataFrame

temp_table_name1 = "HVAC_csv"
temp_table_name2 = "building_csv"

df1.createOrReplaceTempView(temp_table_name1)
df2.createOrReplaceTempView(temp_table_name2)


# COMMAND ----------

# MAGIC %md
# MAGIC Print the schema of each DataFrame to understand the structure of your data, which is essential for subsequent data manipulation.

# COMMAND ----------

# Print the schema of the first DataFrame to check column names
df1.printSchema()

# Print the schema of the second DataFrame to check column names
df2.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC List all available tables in your current database session to verify that your views have been created successfully.

# COMMAND ----------

# List all tables in the current database
tables = spark.catalog.listTables()
print(tables)

# COMMAND ----------

# MAGIC %md
# MAGIC Limit the data to the first 10 rows to simplify initial data handling and visualization for **df1** and store it in new dataframe named "**HVAC_df**"

# COMMAND ----------

# Limit df1 to the first 10 rows and store it in HVAC_df
HVAC_df = df1.limit(10)

# Display the limited DataFrame HVAC_df
display(HVAC_df)


# COMMAND ----------

# MAGIC %md
# MAGIC Limit the data to the first 10 rows to simplify initial data handling and visualization for **df2** and store it in new dataframe named "**building_df**" 

# COMMAND ----------

# Limit df2 to the first 10 rows and store it in building_df
building_df = df2.limit(10)

# Display the limited DataFrame building_df
display(building_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Then we have added columns ,`temp_diff`,`temprange`,`extremetemp` columns. This is to enhance dataframe by adding calculated columns and applying conditional logic to categorize data.

# COMMAND ----------

from pyspark.sql.functions import col, when

# Added temp_diff, temprange, and extremetemp columns
HVAC_temperatures= df1.withColumn(
    "temp_diff", col("TargetTemp") - col("ActualTemp")
).withColumn(
    "temprange", when((col("TargetTemp") - col("ActualTemp")) > 5, "COLD")
    .when((col("TargetTemp") - col("ActualTemp")) < -5, "HOT")
    .otherwise("NORMAL")
).withColumn(
    "extremetemp", when((col("TargetTemp") - col("ActualTemp")) > 5, 1)
    .when((col("TargetTemp") - col("ActualTemp")) < -5, 1)
    .otherwise(0)
)

# Display the updated DataFrame HVAC_temperatures_df
display(HVAC_temperatures)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we have to **select** the columns *( "TargetTemp", "ActualTemp", "System", "SystemAge", "temp_diff", "temprange", "extremetemp")* from the dataframe to focus on specific columns of interest to prepare for more **targeted analysis** or **reporting**.

# COMMAND ----------

# Select the required columns from the DataFrame
selected_HVAC_df = HVAC_temperatures.select(
    "TargetTemp", "ActualTemp", "System", "SystemAge", "temp_diff", "temprange", "extremetemp"
)

# Show the first 10 rows of the selected columns
display(selected_HVAC_df.limit(10))


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Perform Data Join**
# MAGIC
# MAGIC - To enhance our analysis and gain more comprehensive insights, we'll combine data from the HVAC systems with building information. 
# MAGIC - This step is critical for contextualizing temperature data within the specifics of each building, such as location and management details.
# MAGIC
# MAGIC **Join HVAC and Building Data**
# MAGIC - We will perform a left join between the `HVAC_temperatures` DataFrame and the `building_df` DataFrame. This join will align each HVAC record with its corresponding building data based on the `BuildingID`.
# MAGIC - After that we will select the columns from both dataframes to display the result.

# COMMAND ----------

# Perform the left join between HVAC_temperatures and building_df and select the required columns
hvac_building = HVAC_temperatures.join(
    building_df,
    HVAC_temperatures.BuildingID == building_df.BuildingID,
    "left"
).select(
    HVAC_temperatures.TargetTemp,
    HVAC_temperatures.ActualTemp,
    HVAC_temperatures.System,
    HVAC_temperatures.SystemAge,
    HVAC_temperatures.temp_diff,
    HVAC_temperatures.temprange,
    HVAC_temperatures.extremetemp,
    building_df.Country,
    building_df.HVACproduct,
    building_df.BuildingMgr
).limit(20)

# Display the resulting DataFrame
display(hvac_building)


