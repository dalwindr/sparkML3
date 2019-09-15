package com.packt.scalada.learning

import ML_scalaAdvanceMethods._
import org.apache.log4j._
import java.lang.System
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.SparkConf

import org.apache.spark.ml.regression.{LinearRegression,GeneralizedLinearRegression,LinearRegressionModel}
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator,Normalizer}
import org.apache.spark.ml.{Pipeline,Transformer,PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator

object MultiClassDataSetMultiClassAlgoWineQualityDF extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = getSparkSessionConnection("MultiClassDataSetMultiClassAlgo")
  
  val mySchema = StructType(Array(
                                    StructField("fixed_acidity",DoubleType,true),
                                    StructField("volatile_acidity" ,DoubleType,true),
                                    StructField("citric_acid", DoubleType,true),
                                    StructField("residual_sugar" ,DoubleType,true),
                                    StructField("chlorides", DoubleType,true),
                                    StructField("free_sulfur_dioxide" ,DoubleType,true),
                                    StructField("total_sulfur_dioxide" ,DoubleType,true),
                                    StructField("density" ,DoubleType,true),
                                    StructField("pH" ,DoubleType,true),
                                    StructField("sulphates" ,DoubleType,true),
                                    StructField("alcohol", DoubleType,true),
                                    StructField("quality", DoubleType,true)
                                    ))
  val rawDF = spark.read.format("csv").
              schema(mySchema).
              option("delimiter",";").
              load("/Users/keeratjohar2305/Downloads/ScalaDataAnalysisCookbook/chapter5-learning/winequality-red.csv").
                    repartition(2).
                    withColumnRenamed("quality", "label")
 

   rawDF.show(40)
   rawDF.printSchema
   summaryCustomized(rawDF).show()
   
   val catColumnList = Array("quality")
   val labelCol= "quality"
   
   val Array(traningDF,testingDF) = rawDF.randomSplit(Array(0.7,0.3),seed=9999)
   println("\n--------------training Data Analysys")
   ML_scalaAdvanceMethods.univariateAnalysis(traningDF, catColumnList )
   ML_scalaAdvanceMethods.dsShape(traningDF)
   ML_scalaAdvanceMethods.dataFitnessCheck(traningDF)
   
   println("\n---------------testing Data Analysys")
   ML_scalaAdvanceMethods.univariateAnalysis(testingDF, catColumnList )
   ML_scalaAdvanceMethods.dsShape(testingDF)
   ML_scalaAdvanceMethods.dataFitnessCheck(testingDF)
   
   //https://github.com/PacktPublishing/Scala-and-Spark-for-Big-Data-Analytics
   
   val featureNumericalCol = rawDF.columns.diff(Array("label"))
   val featureStringCol = Seq()
   
  
}