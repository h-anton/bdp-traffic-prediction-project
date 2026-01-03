import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}

import DatasetReader._

object Main {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()

    // If no master is set (e.g., via spark-submit), default to local[*]
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
    }

    val spark = SparkSession.builder
      .appName("Project Part 1")
      .config(conf)
      .getOrCreate();

    spark.sparkContext.setLogLevel("WARN")

    /* 
     * Task 1.1: Reading data
     */
    val dataset = new DatasetReader(spark, 5000, 100)
    //val dataset = new DatasetReader(spark, 10000, 1000)
    
    val speedDF = dataset.getSpeedDataFrame()
    val volumeDF = dataset.getVolumeDataFrame()
    val staticFeaturesDF = dataset.getStaticFeaturesDataFrame()

    // Optional, part of task 2: remove data that will not be used for increased performance
    val usefulStaticFeaturesDF = staticFeaturesDF.select("speed_limit").persist()

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
    val staticFeaturesNodeIDDF = usefulStaticFeaturesDF
      .withColumn("node", concat(lit("node_"), monotonically_increasing_id() + 1))
      .withColumn("node_id", monotonically_increasing_id() + 1)

    // Convert speed_limit to km/h, that way all measurments are in the same unit => possibly better for ML
    val staticFeaturesNodeIDSpeedLimitDF = staticFeaturesNodeIDDF.withColumn("speed_limit_kmh",
      when(col("speed_limit") === 0, lit(Double.NaN))
        .when(col("speed_limit") === 1, lit(5))
        .when(col("speed_limit") === 2, lit(20))
        .when(col("speed_limit") === 3, lit(30))
        .when(col("speed_limit") === 4, lit(40))
        .when(col("speed_limit") === 5, lit(50))
        .when(col("speed_limit") === 6, lit(60))
        .when(col("speed_limit") === 7, lit(70))
        .when(col("speed_limit") === 8, lit(80))
        .when(col("speed_limit") === 9, lit(90))
        .when(col("speed_limit") === 10, lit(100))
        .when(col("speed_limit") === 11, lit(110))
    )
 
    // join the 3 dataframes
    val trafficDF = longSpeedDF
      .join(longVolumeDF, Seq("timestamp", "node"), "inner")
      .join(staticFeaturesNodeIDSpeedLimitDF, Seq("node"), "inner")

    /* 
     * Task 2: Feature selection
     */
     // Add a lag feature
     //https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-window.html
     //https://downloads.apache.org/spark/docs/3.5.1/api/scala/org/apache/spark/sql/functions$.html#lag(e:org.apache.spark.sql.Column,offset:Int,defaultValue:Any,ignoreNulls:Boolean):org.apache.spark.sql.Column
    val lagFeatureWindow = Window
      .partitionBy("node")
      .orderBy(col("timestamp"))
    // add a time_index column to display output chronologically
    val tafficTimeIndexDF = trafficDF
      .withColumn("time_index", row_number().over(lagFeatureWindow) - 1)
    /* In order to predict traffic density in the near future, I assume the model will will need to look at the
     * past measurements, and try to detect a trend (rising, falling). I will provide 5 datapoints for this, which
     * means there will be 6 datapoints (current + 5 past points), giving traffic info about the last 30 min. */
    val windowSize = 5
    val trafficWithLagsDF = (1 to windowSize).foldLeft(tafficTimeIndexDF) { (df, lagIndex) =>
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
    val dateColumn = trafficWithRollingDF("timestamp")
    val trafficWithFeaturesDF = trafficWithRollingDF
      .withColumn("day_of_week", date_format(dateColumn, "E")) // Strings Mon, Tue, ...
      .withColumn("week_of_year", date_format(dateColumn, "D") % 52) // 52 weeks in a year
      .withColumn("month_of_year", date_format(dateColumn, "M").cast("int"))
      .withColumn("hour_of_day", date_format(dateColumn, "H").cast("int"))
      .withColumn("minutes_of_hour", date_format(dateColumn, "m").cast("int"))

    // Convert days of week to ints
    //https://downloads.apache.org/spark/docs/3.5.1/api/scala/org/apache/spark/sql/Column.html#when(condition:org.apache.spark.sql.Column,value:Any):org.apache.spark.sql.Column
    val trafficWithFeaturesNumericalDF =
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

    // add speed in 30 min for data training
    val finalDF = trafficWithFeaturesNumericalDF
      .withColumn(
        "speed_next_30min",
        lead(col("speed"), 6).over(lagFeatureWindow) // 30 min = 6 x 5 min interval
      )

    // Filter columns based on relevance for traffic prediction
    val importantCols = Array(
      "node_id",
      "day_of_week", "week_of_year", "month_of_year", "hour_of_day", "minutes_of_hour",
      "speed", "speed_lag_1","speed_lag_2","speed_lag_3","speed_lag_4","speed_lag_5",
      "speed_roll_mean","speed_roll_std",
      "speed_limit_kmh",
      "speed_next_30min"
    )

    // Convert to vector
    val assembler = new VectorAssembler()
      .setInputCols(importantCols)
      .setOutputCol("features")

    val processedDF = assembler.transform(finalDF.na.drop()).persist()

    /*
     * Task 3: Using a predictive model
     */
     println("Start training...")
    //https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-regression
    // Split data in test and training data
    val Array(trainingData, testData) = processedDF.randomSplit(Array(0.8, 0.2), seed = 42)


    val gbt = new GBTRegressor()
      .setLabelCol("speed_next_30min")
      .setFeaturesCol("features")
      .setMaxIter(5)
      .setMaxDepth(5) // Default value in Python docs
      .setStepSize(0.01)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(2) // we only have continuous data
      .fit(processedDF)

    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, gbt))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    predictions.select("prediction", "speed_next_30min", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("speed_next_30min")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    /* Results:
     *    5000_100: RSE = 0.8068831809692809, t = 1:45
     *    10000_1000: RSE = 0.8804960008078757, t = 10:35
     */

    println("Predictions for the next 30 min:")
    println("========")

    //val output = predictions
    //  .orderBy("month_of_year", "week_of_year", "day_of_week", "hour_of_day", "minutes_of_hour")
    //  .select("prediction", "node_id", "month_of_year", "week_of_year", "day_of_week", "hour_of_day", "minutes_of_hour")
    //  .collect()
    //https://www.mungingdata.com/apache-spark/aggregations/
    //https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/functions$.html#collect_list(columnName:String):org.apache.spark.sql.Column
    //https://stackoverflow.com/questions/53927574/how-to-round-decimal-in-scala-spark
    val outputDF = predictions
      .withColumn("speed_next_30min", round(col("speed_next_30min"), 2))
      .select("time_index", "node_id", "speed_next_30min")
      .orderBy("time_index", "node_id")
      .groupBy("time_index")
      .agg(collect_list("speed_next_30min").alias("speed_prediction"))
      .select("speed_prediction")
      .foreach { row => // row is [ArraySeq(speed1, speed2, ...)]
        println(row.getSeq(0).mkString(", "))
      }

    model.write.overwrite().save(dataset.getOutputFileName())

    spark.stop()
  }
}
