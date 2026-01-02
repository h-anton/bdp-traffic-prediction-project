import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature.VectorAssembler

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

    /* 
     * Task 1.2: Manipulating data
     */
    // Unpivot speedDF (convert dataframe from wide format to long format)
    // Python docs, but at least it has an example that compiles:
    // https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.unpivot.html
    val timestampColSpeed = Array(speedDF("timestamp"))
    val nodeColsSpeed = speedDF.columns.filter(_.startsWith("node_")).map(col)
    val longSpeedDF = speedDF.unpivot(timestampColSpeed, nodeColsSpeed, "node", "speed")
    // Unpivot volumeDF
    val timestampColVolume = Array(volumeDF("timestamp"))
    val nodeColsVolume = volumeDF.columns.filter(_.startsWith("node_")).map(col)
    val longVolumeDF = volumeDF.unpivot(timestampColVolume, nodeColsVolume, "node", "volume")

    // create a column in staticFeaturesNode with node_id
    //https://downloads.apache.org/spark/docs/3.5.1/api/scala/org/apache/spark/sql/functions$.html#monotonically_increasing_id():org.apache.spark.sql.Column
    val staticFeaturesNodeIDDF = staticFeaturesDF
      .withColumn("node", concat(lit("node_"), monotonically_increasing_id() + 1))
      .withColumn("node_id", monotonically_increasing_id() + 1)
 
    // join the 3 dataframes
    val trafficDF = longSpeedDF
      .join(longVolumeDF, Seq("timestamp", "node"), "full")
      .join(staticFeaturesNodeIDDF, Seq("node"), "inner")

    /* 
     * Task 2: Feature selection
     */
     // Add a lag feature
     //https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-window.html
     //https://downloads.apache.org/spark/docs/3.5.1/api/scala/org/apache/spark/sql/functions$.html#lag(e:org.apache.spark.sql.Column,offset:Int,defaultValue:Any,ignoreNulls:Boolean):org.apache.spark.sql.Column
    val lagFeatureWindow = Window
      .partitionBy("node")
      .orderBy(col("timestamp"))
    /* In order to predict traffic density in the near future, I assume the model will will need to look at the
     * past measurements, and try to detect a trend (rising, falling). I will provide 5 datapoints for this, which
     * means there will be 6 datapoints (current + 5 past points), giving traffic info about the last 30 min. */
    val windowSize = 5
    val trafficWithLagsDF = (1 to windowSize).foldLeft(trafficDF) { (df, lagIndex) =>
      df
        .withColumn(s"speed_lag_$lagIndex", lag(col("speed"), lagIndex).over(lagFeatureWindow))
        .withColumn(s"volume_lag_$lagIndex", lag(col("volume"), lagIndex).over(lagFeatureWindow))
    }

    // Add a rolling statistic feature
    val rollingWindow = lagFeatureWindow.rowsBetween(-windowSize, 0)

    val trafficWithRollingDF = trafficWithLagsDF
      .withColumn("speed_roll_mean", avg(col("speed")).over(rollingWindow))
      .withColumn("volume_roll_mean", avg(col("volume")).over(rollingWindow))
      .withColumn("speed_roll_std", stddev(col("speed")).over(rollingWindow))
      .withColumn("volume_roll_std", stddev(col("volume")).over(rollingWindow))

    // Augment data with day of the week, week of the year and month of the year, as suggested in section D of https://arxiv.org/pdf/2510.02278
    //https://downloads.apache.org/spark/docs/3.5.1/api/scala/org/apache/spark/sql/functions$.html#date_format(dateExpr:org.apache.spark.sql.Column,format:String):org.apache.spark.sql.Column
    //https://spark.apache.org/docs/latest/sql-ref-datetime-pattern.html
    //val dateColumn = trafficWithRollingDF.columns.filter(_.equals("timestamp")).map(col)(0)
    val dateColumn = trafficWithRollingDF("timestamp")
    val trafficWithFeaturesDF = trafficWithRollingDF
      .withColumn("day_of_week", date_format(dateColumn, "E")) // Strings Mon, Tue, ...
      .withColumn("week_of_year", date_format(dateColumn, "D") % 52)
      .withColumn("month_of_year", date_format(dateColumn, "M").cast("int"))
      .withColumn("hour_of_day", date_format(dateColumn, "H").cast("int"))
      .withColumn("minutes_of_hour", date_format(dateColumn, "m").cast("int"))

    // Convert days of week to ints
    //https://downloads.apache.org/spark/docs/3.5.1/api/scala/org/apache/spark/sql/Column.html#when(condition:org.apache.spark.sql.Column,value:Any):org.apache.spark.sql.Column
    val finalDF =
      trafficWithFeaturesDF.withColumn(
        "day_of_week",
        when(col("day_of_week") === "Mon", 1)
          .when(col("day_of_week") === "Tue", 2)
          .when(col("day_of_week") === "Wed", 3)
          .when(col("day_of_week") === "Thu", 4)
          .when(col("day_of_week") === "Fri", 5)
          .when(col("day_of_week") === "Sat", 6)
          .when(col("day_of_week") === "Sun", 7)
          .otherwise(-1)
      )

    // Filter columns based on relevance for traffic prediction
    val importantCols = Array(
      "node_id",
      "day_of_week", "week_of_year", "month_of_year", "hour_of_day", "minutes_of_hour",
      "speed_lag_1","speed_lag_2","speed_lag_3","speed_lag_4","speed_lag_5",
      "volume_lag_1","volume_lag_2","volume_lag_3","volume_lag_4","volume_lag_5",
      "speed_roll_mean","speed_roll_std",
      "volume_roll_mean","volume_roll_std",
      "category",
      "edge_type",
      "speed_limit",
      "region_id",
      "num_segments"
    )

    // Convert to vector
    val assembler = new VectorAssembler()
      .setInputCols(importantCols)
      .setOutputCol("features")

    val processedDF = assembler.transform(finalDF)


    spark.stop()
  }
}
