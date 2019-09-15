package com.packt.scalada.learning
import ML_scalaAdvanceMethods._
import org.apache.log4j._
import java.lang.System
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.SparkConf

//vectorizing
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator,Normalizer}
// text analsysys
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.feature.IDF
object churnBigml_LR {
      Logger.getLogger("org").setLevel(Level.ERROR)
   val spark = getSparkSessionConnection("MultiClassDataSetMultiClassAlgoPenBasedDF")
     val rawDF = spark.read.format("csv").
              //schema(mySchema).
              option("header","true").
              option("inferSchema","true").
              option("delimiter",",").
              load("/Users/keeratjohar2305/Downloads/Dataset/churn-bigml-80.csv")
   val Array(traningDF,testingDF) = rawDF.randomSplit(Array(0.7,0.3),seed=99999999999999999L)
   //val df_subset = rawDF.randomSplit(Array(0.00000001, 0.01), seed = 12345)(0)
   println("sliting doen")
   //println("data "+ testingDF.count())
   traningDF.show(20,false)
   ML_scalaAdvanceMethods.dsShape(traningDF)
   ML_scalaAdvanceMethods.summaryCustomized(traningDF).show()
  
}