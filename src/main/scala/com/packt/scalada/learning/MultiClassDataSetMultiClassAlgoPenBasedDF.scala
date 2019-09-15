package com.packt.scalada.learning

import ML_scalaAdvanceMethods._
import org.apache.log4j._
import java.lang.System
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.SparkConf


  object MultiClassDataSetMultiClassAlgoPenBasedDF extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = getSparkSessionConnection("MultiClassDataSetMultiClassAlgoPenBasedDF")
  
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
              //schema(mySchema).
              option("delimiter",",").
              load("/Users/keeratjohar2305/Downloads/Dataset/penbased.dat").
                    repartition(6)
                    //.
                   // withColumnRenamed("quality", "label")
  
  val oldColumnName= rawDF.columns             
  val newColumnName= rawDF.columns.map(x=> "pix" + x.drop(1))
  dsShape(rawDF)
  
  val newDF = //rawDF(columnName :_*)
        rawDF.
        withColumn(newColumnName(0),col(oldColumnName(0)).cast(DoubleType)).drop(col(oldColumnName(0))).
        withColumn(newColumnName(1),col(oldColumnName(1)).cast(DoubleType)).drop(col(oldColumnName(1))).
        withColumn(newColumnName(2),col(oldColumnName(2)).cast(DoubleType)).drop(col(oldColumnName(2))).
        withColumn(newColumnName(3),col(oldColumnName(3)).cast(DoubleType)).drop(col(oldColumnName(3))).
        withColumn(newColumnName(4),col(oldColumnName(4)).cast(DoubleType)).drop(col(oldColumnName(4))).
        withColumn(newColumnName(5),col(oldColumnName(5)).cast(DoubleType)).drop(col(oldColumnName(5))).
        withColumn(newColumnName(6),col(oldColumnName(6)).cast(DoubleType)).drop(col(oldColumnName(6))).
        withColumn(newColumnName(7),col(oldColumnName(7)).cast(DoubleType)).drop(col(oldColumnName(7))).
        withColumn(newColumnName(8),col(oldColumnName(8)).cast(DoubleType)).drop(col(oldColumnName(8))).
        withColumn(newColumnName(9),col(oldColumnName(9)).cast(DoubleType)).drop(col(oldColumnName(9))).
        withColumn(newColumnName(10),col(oldColumnName(10)).cast(DoubleType)).drop(col(oldColumnName(10))).
        withColumn(newColumnName(11),col(oldColumnName(11)).cast(DoubleType)).drop(col(oldColumnName(11))).
        withColumn(newColumnName(12),col(oldColumnName(12)).cast(DoubleType)).drop(col(oldColumnName(12))).
        withColumn(newColumnName(13),col(oldColumnName(13)).cast(DoubleType)).drop(col(oldColumnName(13))).
        withColumn(newColumnName(14),col(oldColumnName(14)).cast(DoubleType)).drop(col(oldColumnName(14))).
        withColumn(newColumnName(15),col(oldColumnName(15)).cast(DoubleType)).drop(col(oldColumnName(15))).
        withColumn("label",col(oldColumnName(16)).cast(DoubleType)).drop(col(oldColumnName(16)))
        
    //newDF.show()   
 
  

   newDF.show(40)
   newDF.printSchema
   
   summaryCustomized(newDF).show()
   ML_scalaAdvanceMethods.dsShape(newDF)
   val catColumnList = Array("label")
   val labelCol= "label"

   val Array(traningDF,testingDF) = newDF.randomSplit(Array(0.7,0.3),seed=9999)
   println("\n--------------training Data Analysys")
   ML_scalaAdvanceMethods.univariateAnalysis(traningDF, catColumnList )
   ML_scalaAdvanceMethods.dsShape(traningDF)
   ML_scalaAdvanceMethods.dataFitnessCheck(traningDF)
   
   println("\n---------------testing Data Analysys")
   ML_scalaAdvanceMethods.univariateAnalysis(testingDF, catColumnList )
   ML_scalaAdvanceMethods.dsShape(testingDF)
   ML_scalaAdvanceMethods.dataFitnessCheck(testingDF)
   
   //https://github.com/PacktPublishing/Scala-and-Spark-for-Big-Data-Analytics
   val featureStringCol = Seq()
   val featureNumericalCol = newDF.columns.diff(Array("label")).toSeq
   
   ML_scalaAdvanceMethods.CallOneVsALLFullAlgo(traningDF,testingDF,"MultiClass",featureStringCol,featureNumericalCol)
   
}