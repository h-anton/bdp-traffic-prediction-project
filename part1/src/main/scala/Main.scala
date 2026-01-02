import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import DatasetReader._

object Main {

  def main(args: Array[String]): Unit = {

    // Create Spark session
    val spark = SparkSession.builder()
      .appName("Project Part 1")
      .master("spark://spark-master:7077")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    /* 
     * Task 1.1: Reading data
     */
    val dataset = new DatasetReader(spark, 5000, 100)
    val speedDF = dataset.getSpeedDataFrame()
    val volumeDF = dataset.getVolumeDataFrame()
    val staticFeaturesDF = dataset.getStaticFeaturesDataFrame()

    // Show small section of data
    val schemaListSpeed: List[String] = speedDF.columns.toList
    speedDF.select(schemaListSpeed(0), schemaListSpeed(1), schemaListSpeed(2)).show(5)
    val schemaListVolume: List[String] = volumeDF.columns.toList
    volumeDF.select(schemaListVolume(0), schemaListVolume(1), schemaListVolume(2)).show(5)
    staticFeaturesDF.printSchema()
    val schemaListFeatures: List[String] = staticFeaturesDF.columns.toList
    staticFeaturesDF.select(schemaListFeatures(0), schemaListFeatures(1), schemaListFeatures(2)).show(5)

    /* 
     * Task 1.2: Manipulating data
     */
    // Unpivot speedDF (convert dataframe from wide format to long format)
    // Python docs, but at least it has an example that compiles:
    // https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.unpivot.html
    val timestampColSpeed = speedDF.columns.filter(_.equals("timestamp")).map(col)
    val nodeColsSpeed = speedDF.columns.filter(_.startsWith("node_")).map(col)
    val longSpeedDF = speedDF.unpivot(timestampColSpeed, nodeColsSpeed, "node", "speed")
    // Unpivot volumeDF
    val timestampColVolume = volumeDF.columns.filter(_.equals("timestamp")).map(col)
    val nodeColsVolume = volumeDF.columns.filter(_.startsWith("node_")).map(col)
    val longVolumeDF = volumeDF.unpivot(timestampColVolume, nodeColsVolume, "node", "volume")

    // create a column in staticFeaturesNode with node_id
    //https://downloads.apache.org/spark/docs/3.5.1/api/scala/org/apache/spark/sql/functions$.html#monotonically_increasing_id():org.apache.spark.sql.Column
    val staticFeaturesNodeIDDF = staticFeaturesDF.withColumn("node", concat(lit("node_"), monotonically_increasing_id() + 1))
 
    // join the 3 dataframes
    val trafficDF = longSpeedDF
      .join(longVolumeDF, Seq("timestamp", "node"), "full")
      .join(staticFeaturesNodeIDDF, Seq("node"), "inner")

    // Show new schema
    trafficDF.printSchema()

    // Check output size to see that no data is lost
    print("trafficDF size: ")
    print(trafficDF.count())
    print(" x ")
    println(trafficDF.columns.length)

    // Sample output
    trafficDF.select("node", "timestamp", "speed", "volume", "category", "edge_type").show(5)



    spark.stop()
  }
}
