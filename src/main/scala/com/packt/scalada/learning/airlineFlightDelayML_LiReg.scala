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
import org.apache.spark.ml.Pipeline
// text analsysys
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.feature.IDF
import org.apache.spark.sql.catalyst.plans.logical.Distinct

object airlineFlightDelayML_LiReg extends App{
    Logger.getLogger("org").setLevel(Level.ERROR)
   val spark = getSparkSessionConnection("MultiClassDataSetMultiClassAlgoPenBasedDF")
   
      val myschema= StructType(Seq(
       StructField("Year",IntegerType,true),
       StructField("Month",IntegerType,true),
       StructField("DayofMonth",IntegerType,true),
       StructField("DayOfWeek",IntegerType,true),
       
       StructField("DepTime",IntegerType,true),
       StructField("CRSDepTime",IntegerType,true),
       StructField("ArrTime",IntegerType,true),
       StructField("CRSArrTime",IntegerType,true),
       
       StructField("UniqueCarrier",StringType,true),
       
       StructField("FlightNum",IntegerType,true),
       StructField("TailNum",StringType,true),
       
       StructField("ActualElapsedTime",IntegerType,true),
       StructField("CRSElapsedTime",IntegerType,true),
       StructField("AirTime",IntegerType,true),
       StructField("ArrDelay",IntegerType,true),
       StructField("DepDelay",IntegerType,true),
       
       StructField("Origin",StringType,true),
       StructField("Dest",StringType,true),
       
       StructField("Distance",IntegerType,true),
       StructField("TaxiIn",IntegerType,true),
       StructField("TaxiOut",IntegerType,true),
       StructField("Cancelled",IntegerType,true),
       StructField("CancellationCode",IntegerType,true),
       
       StructField("Diverted",IntegerType,true),
       StructField("CarrierDelay",IntegerType,true),
       StructField("WeatherDelay",IntegerType,true),
       StructField("NASDelay",IntegerType,true),
       StructField("SecurityDelay",IntegerType,true),
       StructField("LateAircraftDelay",IntegerType,true)
      ))
    import com.databricks.spark.csv
    val rawDF = spark.read.format("com.databricks.spark.csv").
              //schema(myschema).
              option("header","true").
              option("inferSchema","true").
              option("delimiter",",").
              load("/Users/keeratjohar2305/Downloads/Dataset/airlineFlightDelaySample.csv").drop("CancellationCode").drop("Year").drop("FlightNum").drop("TailNum").
     
              withColumn("AirTime", col("AirTime").cast(IntegerType)).
              //withColumn("ArrDelay", col("ArrDelay").cast(StringType)).
              withColumn("DepDelay", col("DepDelay").cast(IntegerType)).
              withColumn("ActualElapsedTime", col("ActualElapsedTime").cast(IntegerType))
              //withColumnRenamed("ArrDelay", "label")
              
   rawDF.printSchema()
   rawDF.show(20,false)
   
   val Array(traningDF1,testingDF1) = rawDF.randomSplit(Array(0.7,0.3),seed=9999999999L)
   
   val traningDF= new StringIndexer().setInputCol("ArrDelay").setOutputCol("label").fit(traningDF1).transform(traningDF1)
   val testingDF= new StringIndexer().setInputCol("ArrDelay").setOutputCol("label").fit(testingDF1).transform(testingDF1)
   
   println("traininDF label count " + traningDF.select("label").distinct().count())
    println("testingDF label count " + testingDF.select("label").distinct().count())
    val t1= traningDF.select("label").distinct()
    val t2 =  testingDF.select("label").distinct()    
    println(t1.join(t2, Seq("label")).filter(t1.col("label")=== t2.col("label")).collect().toList)
    println(t1.join(t2, Seq("label")).filter(t1.col("label") !== t2.col("label")).collect().toList)
    testingDF.show(20,false)

   
//   println("sliting doen")
//   traningDF.show(20,false)
//   ML_scalaAdvanceMethods.dsShape(traningDF)
     ML_scalaAdvanceMethods.summaryCustomized(traningDF).show()
//   traningDF.printSchema()
//   
   import org.apache.spark.ml.classification.LogisticRegression
   import org.apache.spark.ml.feature.Binarizer
   import org.apache.spark.ml.feature.VectorSlicer
   import org.apache.spark.ml.Pipeline
   import org.apache.spark.ml.feature.StandardScaler

 //assemble raw feature



    val labelCol = Array("label")
    val catgoricalCatColumn = Seq("Month","DayofMonth","DayOfWeek","UniqueCarrier","Origin","Dest","Diverted","Cancelled")
    val NumericalCatColumns = Seq( "CRSDepTime", "CRSArrTime", "ActualElapsedTime", "CRSElapsedTime", "DepDelay", "Distance")
    
    // Pipeline creatioon from { assembler,slicer,scaler,binarizerClassifier,lrPipeline}
    val assembler = new VectorAssembler().setInputCols(catgoricalCatColumn.map(_+"Indexed").toArray ++ NumericalCatColumns).setOutputCol("features").setHandleInvalid("skip")
    
    //val slicer = new VectorSlicer().setInputCol("rawFeatures").setOutputCol("slicedfeatures").setNames(catgoricalCatColumn.map(_+"Indexed").toArray ++ NumericalCatColumns)
    //val scaler = new StandardScaler().setInputCol("slicedfeatures").setOutputCol("features").setWithStd(true).setWithMean(true)
   // val binarizerClassifier = new Binarizer().setInputCol("ArrDelay").setOutputCol("label").setThreshold(15.0)
     //logistic regression
     val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setLabelCol("label").setFeaturesCol("features")

    // Chain indexers and tree in a Pipeline
    val lrPipeline = new Pipeline().setStages(getStringIndexersArray(catgoricalCatColumn)++ Array(assembler))



      // Train model. 
      val pipedDF = lrPipeline.fit(traningDF).transform(traningDF)
      // Make predictions.
      
      val testPipedDF = lrPipeline.fit(testingDF).transform(testingDF)
      // Select example rows to display.
     
      CallOneVsALLAlgo(pipedDF,testPipedDF,"MultiClass")
         
}