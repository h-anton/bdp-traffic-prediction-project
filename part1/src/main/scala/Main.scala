import org.apache.spark.sql.SparkSession

import DatasetReader._

object Main {

  def main(args: Array[String]): Unit = {

    // Create Spark session
    val spark = SparkSession.builder()
      .appName("Project Part 1")
      .master("spark://spark-master:7077")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    // Read data
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

    spark.stop()
  }
}
