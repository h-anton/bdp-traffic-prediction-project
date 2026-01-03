
/***** DEFINE THESE PARAMS *****/
val projectName = "hello-spark-streaming-kafka"
val mainClassName = "Main"
val outputJarName = "myProject.jar"
/*******************************/

val scalaVer = "2.13.16"
val sparkVer = "4.0.0"
val kafkaVer = "4.0.0"
val log4JVer = "2.24.1"

name := projectName
version := "0.1-SNAPSHOT"
organization := "org.anton"

ThisBuild / scalaVersion := scalaVer

val sparkDependencies = Seq(
      "org.apache.spark" %% "spark-sql" % sparkVer % "provided",
      "org.apache.spark" %% "spark-streaming" % sparkVer % "provided"
      )


val kafkaDependencies = Seq(
      // "org.apache.kafka" % "kafka-clients" % kafkaVer % "provided",  // this does not work: the correct `kafka-clients` dependency (3.9.0?) needs to be pulled in through `spark-sql-kafka-0-10` for `sbt run`
      "org.apache.logging.log4j" % "log4j-slf4j-impl" % log4JVer % "provided"
      )

lazy val root = (project in file(".")).
  settings(
    libraryDependencies ++= sparkDependencies ++ kafkaDependencies,
    libraryDependencies += "org.apache.spark" %% "spark-sql-kafka-0-10" % sparkVer % "provided" // spark/kafka bridge
  )

assembly / mainClass := Some(mainClassName)
assembly / assemblyJarName := outputJarName

// make run command include the provided dependencies
Compile / run  := Defaults.runTask(Compile / fullClasspath,
  Compile / run / mainClass,
  Compile / run / runner
).evaluated

// stays inside the sbt console when we press "ctrl-c" while a Spark program executes with "run" or "runMain"
Compile / run / fork := true
Global / cancelable := true

// exclude Scala library from assembly
assembly / assemblyOption := (assembly / assemblyOption).value.withIncludeScala(false)
