
/***** DEFINE THESE PARAMS *****/
val projectName = "part1"
val mainClassName = "Main"
val outputJarName = "myProject.jar"
/*******************************/

val scalaVer = "2.13.16"
val sparkVer = "4.0.0"

name := projectName
version := "0.1-SNAPSHOT"
organization := "org.anton"

ThisBuild / scalaVersion := scalaVer

val sparkDependencies = Seq(
      "org.apache.spark" %% "spark-sql" % sparkVer % "provided",
      "org.apache.spark" %% "spark-mllib" % sparkVer % "provided",
      )

lazy val root = (project in file(".")).
  settings(
    libraryDependencies ++= sparkDependencies
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
