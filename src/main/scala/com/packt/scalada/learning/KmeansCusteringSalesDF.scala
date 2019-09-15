package com.packt.scalada.learning


import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.SparkConf

import org.apache.spark.ml.clustering.{ KMeans=> KMeansML}
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator}
import org.apache.spark.ml.{Pipeline,Transformer,PipelineModel}
import org.apache.log4j._
import java.lang.System

import ML_scalaAdvanceMethods._
object KmeansCusteringSalesDF extends App{
  
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = ML_scalaAdvanceMethods.getSparkSessionConnection("TEst App")
    import spark.sqlContext.implicits._


     val rawDF = spark.read.format("csv").option("header", "true").
                    option("inferSchema","true").
                    option("delimiter",",").
                    load("/Users/keeratjohar2305/Downloads/ScalaDataAnalysisCookbook/chapter5-learning/sales_train.csv").
                    repartition(2)
 
  
println("\n***********   START: sales Data Exploration and missing Values Replacement *****************\n")






rawDF.show(10)
rawDF.printSchema
rawDF.columns
println("\nSales Data Summary ")  
summaryCustomized(rawDF).show()
//rawDF.summary().show()


// Print Data from categrical columns
println ("\nFrequency of Categories for categorical variables:-")

rawDF.columns.foreach{ fieldName =>
       if ( Seq("Item_Fat_Content","Item_Type","Outlet_Identifier","Outlet_Establishment_Year","Outlet_Size","Outlet_Location_Type","Outlet_Type").
                        contains(fieldName)
          )
              println(fieldName,":",rawDF.groupBy(fieldName).count().collect().toList)
      else
              None
}

     
 // 1) missing value Fill for numeric continues column
val rawDFMissingFilled  = missingValFilled(rawDF,"Item_Weight")  
 // rawDFCleaned.na.replace("Outlet_Size", )
  //summaryCustomized(rawDFMissingFilled).show()

 //2) Missing values filling based one column based on another
val myFunct = udf((str: String)=> if (str!=null && Array("Supermarket Type1",  "Supermarket Type3" ,"Supermarket Type2" ,"Grocery Store").contains(str) ) 
                                  Map("Supermarket Type1"-> "High", "Supermarket Type3"-> "small" ,  "Supermarket Type2"-> "small","Grocery Store"-> "Mediaum")(str)
                                  else ""
                  )             
spark.udf.register("myFunct", myFunct)

val NewDF= rawDFMissingFilled.withColumn("Outlet_Size1",when (col("Outlet_Size").isNull, myFunct(col("Outlet_Type"))).otherwise(col("Outlet_Size")))
//val mapOutlet_Type = Map("Supermarket Type1"-> "High", "Supermarket Type3"-> "small" ->  "Supermarket Type2"-> "small","Grocery Store,"-> "Mediaum")
    
summaryCustomized(NewDF).show()



val Array(traningDF,testingDF) = NewDF.randomSplit(Array(0.7,0.3),seed=99999999)




println("\n***********   END: sales Data Exploration and missing Values Replacement *****************\n")

println("\n***********   Start: sales Data ML piple Creation *****************\n")





// Seperate out String feature column and numeric features
 val featuresCatColNames = Seq("Item_Identifier", "Item_Fat_Content", "Item_Type","Outlet_Identifier","Outlet_Size1","Outlet_Location_Type","Outlet_Type")
 val featuresNumColNames = Seq("Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year","Item_Outlet_Sales")
 

 val stages = CategoricalFeatureVectorzing(featuresCatColNames) ++ FeatureAssembler(featuresCatColNames,featuresNumColNames )
  
// Setup k-means model with two clusters
val pipelineInstatiated = new Pipeline().setStages(stages)

val pipedtrainingDF = pipelineInstatiated.fit(traningDF).transform(traningDF)
println("pipedDF")
pipedtrainingDF.show(false)


println("\n***********   End: sales Data ML piple Creation *****************\n")



println("trained DF")

 //Lets train K means model
val MLkmeans = new KMeansML().setK(10).setSeed(1L)
val MLkmeansTrainedDF = MLkmeans.fit(pipedtrainingDF)
val Kmean_prediction =  MLkmeansTrainedDF.transform(pipedtrainingDF)

 //Kmean_prediction.select(col("prediction")).distinct().collect().map(_.mkString.toInt)
 import spark.sqlContext.implicits._
 
 (0 to 9).foreach(x=>
 Kmean_prediction.filter(col("prediction")===x).show(4)
 )
 


//lets make the pridiction kmeansPipeline
// create piped test DF
 //val pipedtestDF = pipelineInstatiated.fit(testingDF).transform(testingDF)
 //println("pipedtestDF testing DF")
 //pipedtestDF.show(false)
 
 

  
}