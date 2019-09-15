organization := "com.packt"

name := "chapter4-learning"

scalaVersion := "2.11.12"
val sparkVersion="2.4.0"

libraryDependencies  ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.databricks" %% "spark-csv" % "1.0.3",
  "org.scalanlp" %% "epic" % "0.5",
  "org.scalanlp" %% "epic-parser-en-span" % "2016.8.29",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.9.2",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.9.2" classifier "models"
)

fork := true
