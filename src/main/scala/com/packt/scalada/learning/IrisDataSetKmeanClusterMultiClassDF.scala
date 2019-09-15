package com.packt.scalada.learning
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._

import org.apache.log4j._
import org.apache.spark.sql.SparkSession

import java.lang.System
import ML_scalaAdvanceMethods._

object IrisDataSetKmeanClusterMultiClassDF extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
 
  //Frankly, we could make this a tuple but this looks neaton
  case class Document(label: String, content: String)
  val spark = SparkSession
              .builder()
              .appName("Java Spark SQL basic example")
              .config("spark.master", "local")
              .getOrCreate()
   import spark.sqlContext.implicits._
   //DataSet1 
 
 val IrisSchema = StructType(Seq(
                   StructField("sepal_length",DoubleType,true),
                   StructField("sepal_width",DoubleType,true),
                   StructField("petal_length",DoubleType,true),
                   StructField("petal_width",DoubleType,true),
                   StructField("class",StringType,true)))
 
 //DataSet2 in DF format
 val IrisDF = spark.read.format("csv").option("header", "true").
                    schema(IrisSchema).
                    option("delimiter",",").
                    load("/Users/keeratjohar2305/Downloads/ScalaDataAnalysisCookbook/chapter5-learning/iris.data").
                    repartition(2)
println("\nIris DF Data Summary")

 IrisDF.printSchema()
 IrisDF.show(20)
 dsShape(IrisDF)
 //Preliminary EDA
 summaryCustomized(IrisDF).show()
 
 //split( thedata)
   
 
 
 val label_vol = Seq("class")
 val IrisCleanedDF = ML_scalaAdvanceMethods.getStringIndexersArray(label_vol)(0).fit(IrisDF).transform(IrisDF).withColumnRenamed("classIndexed", "label").drop("class")
  
 val Array(traningDF,testingDF) = IrisCleanedDF.randomSplit(Array(0.7,0.3), seed=9999)
   
 val featuresCatColNames = Seq()
 val featuresNumColNames = traningDF.columns.drop(1).toSeq
      
       // pipelining the stages
       val stages = CategoricalFeatureVectorzing(featuresCatColNames) ++ 
                    FeatureAssembler(featuresCatColNames,featuresNumColNames) 
       
                 
      //Something missed ...I Did not considered featuresNumColNames
      // Ok doubt is clear   func( FeatureAssembler ) is processing both numerical and cataogiral features         
                
       // pipelinedStages 
       import org.apache.spark.ml.{Pipeline,Transformer,PipelineModel}
       val pipelinedStages = new Pipeline().setStages(stages)
       
       
       // create piped train DF
       val pipedDF = pipelinedStages.fit(traningDF).transform(traningDF)
       
       println("pipedDF training DF")
       pipedDF.show()
       
       // create piped test DF
       val pipedtestDF = pipelinedStages.fit(testingDF).transform(testingDF)
       println("pipedtestDF testing DF")
       pipedtestDF.show()

       CallNaiveBayesAlgo(pipedDF,pipedtestDF, "MultiClass")
       CallOneVsALLAlgo(pipedDF, pipedtestDF, "MultiClass")
       ///CallGradiantBoosterTreeLAlgo(pipedDF, pipedtestDF) , Only support binary classification
       CallDecisionTreeClassifierLAlgo(pipedDF, pipedtestDF, "MultiClass")
       CallRandomForestClassifierLAlgo(pipedDF, pipedtestDF, "MultiClass")
       CallLogisticRegressionAlgo(pipedDF, pipedtestDF, "MultiClass")
       CallMultiLayerPerceptrolAlgo(pipedDF, pipedtestDF, "MultiClass")
 

}