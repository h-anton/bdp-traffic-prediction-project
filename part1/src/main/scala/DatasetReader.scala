import org.apache.spark.sql.{DataFrame, SparkSession}

class DatasetReader(
    spark: SparkSession,
    measurements: Int,
    segments: Int,
    basePath: String = "../data"
) {

  // Read file
  private def loadFile(fileName: String): DataFrame = {
    val fullPath = s"$basePath/$fileName"
    println(fullPath)
    spark.read.parquet(fullPath)
  }

  // Read Metric
  private def loadMetric(metric: String): DataFrame = {
    val fileName = s"city_M_${metric}_${measurements}_${segments}.parquet"
    loadFile(fileName)
  }

  // Public methods
  def getSpeedDataFrame(): DataFrame = loadMetric("speed")
  def getVolumeDataFrame(): DataFrame = loadMetric("volume")
  def getStaticFeaturesDataFrame(): DataFrame = {
    val fileName = s"city_M_static_features.parquet"
    loadFile(fileName)
  }
  def getOutputFileName(): String = s"file:///workspace/part1/output_${measurements}_${segments}.model"
}