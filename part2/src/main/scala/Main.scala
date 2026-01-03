import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.streaming.{StreamingContext, Seconds}

object Main {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()

    // If no master is set (e.g., via spark-submit), default to local[*]
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
    }

    val spark = SparkSession.builder
      .appName("Project Part 2")
      .config(conf)
      .getOrCreate();

    spark.sparkContext.setLogLevel("WARN")

    // kafka config
    val bootstrapServers = "kafka:9094"

    /*
     * Task 1: Setup a streaming pipeline
     */

    // Speed and Volume topics return data like this:
    // {"timestamp":"2024-07-01T19:05:00","node_id":"node_281","value":1}
    val schema = new StructType()
      .add("timestamp", StringType)
      .add("node_id", StringType)
      .add("value", DoubleType)

    // streams for speed and volume topics from kafka
    val speedKafkaStream = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", bootstrapServers)
      .option("subscribe", "speed")
      .load()

    val volumeKafkaStream = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", bootstrapServers)
      .option("subscribe", "volume")
      .load()

    // Convert JSON to correct schema, add 5 min watermark
    val speedDF = speedKafkaStream
      .selectExpr("CAST(value AS STRING) as json_str")
      .select(from_json(col("json_str"), schema).as("data"))
      .select(
        col("data.timestamp").cast("timestamp").as("timestamp"),
        col("data.node_id").as("node"),
        col("data.value").as("speed")
      )
      .withWatermark("timestamp", "5 minutes")

    val volumeDF = volumeKafkaStream
      .selectExpr("CAST(value AS STRING) as json_str")
      .select(from_json(col("json_str"), schema).as("data"))
      .select(
        col("data.timestamp").cast("timestamp").as("timestamp"),
        col("data.node_id").as("node"),
        col("data.value").as("volume")
      )
      .withWatermark("timestamp", "5 minutes")

    // Join speed and volume dataframes
    val joinedDF = speedDF.join(
      volumeDF,
      speedDF("node") === volumeDF("node") &&
      speedDF("timestamp") === volumeDF("timestamp"),
      "inner"
    )

    // remove duplicates
    val resultDF = joinedDF.select(
      speedDF("timestamp"),
      speedDF("node"),
      speedDF("speed"),
      volumeDF("volume")
    )

    /*
     * Task 2: Recompute features online
     */
    


    // output stream to console
    val query = resultDF.writeStream
      .format("console")
      .outputMode("append")
      .option("truncate", false)
      .start()

    query.awaitTermination()

    //spark.stop()
  }
}