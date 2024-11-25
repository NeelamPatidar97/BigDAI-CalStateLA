
### ✨ **Author Information** 

#|                    |                                                                 |
#|--------------------|-----------------------------------------------------------------|
#| **Author**         | Neelam Patidar                                                  |
#| **Date**           | July 25, 2024                                                 |
#| **Supervisor**     | Prof. Jongwook Woo                                              |
#| **Affiliation**    | Big Data AI Center (BigDAI): High Performance Information Computing Center (HiPIC) |



#### 📚 **Tutorial Overview**

#Many personal and commercial devices now contain sensors that collect information from the physical world, an area often referred to as the Internet of Things (IoT). For example:
#- Most phones have a GPS.
#- Fitness devices track how many steps you've taken.
#- Thermostats can monitor the temperature of a building.

#In this tutorial, you'll learn how Databricks, utilizing its DBFS and PySpark, can be used to process historical data produced by heating, ventilation, and air conditioning (HVAC) systems to identify systems that are not able to reliably maintain a set temperature, a popular IoT data analysis task. You will learn how to:
#- Refine and enrich temperature data from buildings in several countries.
#- Analyze the data to determine which buildings have problems maintaining comfortable temperatures (actual recorded temperature vs. the temperature the thermostat was set to).
#- Infer the reliability of HVAC systems used in the buildings.

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Create or retrieve a Spark session
spark = SparkSession.builder.appName("HVAC Sensor Analysis").getOrCreate()

# Load data from HVAC CSV files stored in HDFS
hvac_df = spark.read.csv("/user/npatida/SensorFiles/hvac", header=True, inferSchema=True)

# Load data from Building CSV files stored in HDFS
building_df = spark.read.csv("/user/npatida/SensorFiles/building", header=True, inferSchema=True)

# Show the first few rows of each DataFrame to verify successful loading
hvac_df.show(10)
building_df.show(10)

# List all tables in the current database and print the details of each table
tables = spark.catalog.listTables()
for table in tables:
    print(f"Name: {table.name}, Database: {table.database}, Is Temporary: {table.isTemporary}")

# Added temp_diff, temprange, and extremetemp columns
from pyspark.sql.functions import col, when
hvac_temperatures= hvac_df.withColumn(
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
hvac_temperatures.show()

# Select the required columns from the DataFrame
selected_hvac_df = hvac_temperatures.select(
    "TargetTemp", "ActualTemp", "System", "SystemAge", "temp_diff", "temprange", "extremetemp"
)
selected_hvac_df.show(10)


# Perform the left join between HVAC_temperatures and building_df and select the required columns
hvac_building = hvac_temperatures.join(
    building_df,
    hvac_temperatures.BuildingID == building_df.BuildingID,
    "left"
).select(
    hvac_temperatures.TargetTemp,
    hvac_temperatures.ActualTemp,
    hvac_temperatures.System,
    hvac_temperatures.SystemAge,
    hvac_temperatures.temp_diff,
    hvac_temperatures.temprange,
    hvac_temperatures.extremetemp,
    building_df.Country,
    building_df.HVACproduct,
    building_df.BuildingMgr
)
hvac_building.show(10)


