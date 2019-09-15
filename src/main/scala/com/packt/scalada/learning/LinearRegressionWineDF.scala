package com.packt.scalada.learning

import ML_scalaAdvanceMethods._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.SparkConf

import org.apache.spark.ml.regression.{LinearRegression,GeneralizedLinearRegression,LinearRegressionModel}
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator,Normalizer}
import org.apache.spark.ml.{Pipeline,Transformer,PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.log4j._
import java.lang.System


object LinearRegressionWineDF extends App {
 
      Logger.getLogger("org").setLevel(Level.ERROR)
    
     val spark = SparkSession
              .builder()
              .appName("Java Spark SQL basic example")
              .config("spark.master", "local")
              .getOrCreate()
    import spark.sqlContext.implicits._
   
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
                    repartition(2)
 

   rawDF.show()
   rawDF.printSchema
   summaryCustomized(rawDF).show()
   
   
   val Array(traningDF,testingDF) = rawDF.randomSplit(Array(0.7,0.3),seed=99999999)

    println("""
    //**************************************************
  
  // The Ist way of performing Linear regression
   using pipeline
  //*************************************************""")

  val featuresCatColNames = Seq()
  val featuresNumColNames = Seq( "fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol")
  val labelCol= "quality"
  
  val featuresAssembler = FeatureAssembler(featuresCatColNames,featuresNumColNames )
 
  
  val labelIndexer = new StringIndexer().setInputCol(labelCol).setOutputCol("label");
   
  val lr = new LinearRegression().setLabelCol(labelCol)
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

  // create pipeline 
  val  pipeline =  new Pipeline().setStages(featuresAssembler   ++ Array(lr))
  
  
  //traning the model
  val TrainedDF = pipeline.fit(traningDF)
 
  
    
   // Model review parameter
   val linRegModel = TrainedDF.stages(1).asInstanceOf[LinearRegressionModel]
   val trainingSummary2 = linRegModel.summary  
 
    
    println(s"Model Coefficients: ${linRegModel.coefficients} Intercept: ${linRegModel.intercept}")    
    println(s"numIterations: ${trainingSummary2.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary2.objectiveHistory.mkString(",")}]")
    trainingSummary2.residuals.show()
    println(s"RMSE: ${trainingSummary2.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary2.r2}")
   
    TrainedDF.transform(testingDF).show()   
  
  
  val predictionDF = TrainedDF.transform(testingDF)
  //val featuredTestingDF = pipeline.fit(traningDF).transform(testingDF)
  
  predictionDF.show()
 
   
  println ("prediction matrix are : ")
  Seq(("mae" ,"Mean Absalue Error"), ( "mse","mean Square Error"), ("rmse","root Mean Squared Error"), ("r2" , "R2"))
              .foreach{  matrixname=>
                  val regEval = new RegressionEvaluator().
                  setMetricName(matrixname._1).
                  setPredictionCol("prediction").
                  setLabelCol(labelCol)
                  println ("prediction matrix :" + matrixname._2 + " , Values is  " + regEval.evaluate(predictionDF))
                }
  //val cnt =  predictionDF.select(abs((col("quality")- col("prediction"))/col("quality"))).count()
  //val sss = predictionDF.select(abs((col("quality")- col("prediction"))/col("quality"))).agg(sum("")).collect()(0).mkString.toDouble
  //println("count " + cnt + "sss " + "median Error " + cnt/sss )
  
  
  
   println("""
  //**************************************************
  
  // The Second way of performing Linear regression
  
  //*************************************************""")
  val newDF = rawDF.withColumn("label", col("quality"))
  
  val featuresAssemblerDF =featuresAssembler(0).transform(newDF)//.select("quality","label","features")
  print("printing features dataset")
  featuresAssemblerDF.show()
 //normalise [9] the data so that we can have a better result
  val noramalizedFeatureDF =  new Normalizer()
                              .setInputCol("features")
                              .setOutputCol("normFeatures")
                              .setP(2.0)
                              .transform(featuresAssemblerDF)
   //Function setP(2.0) [10] has ensured that Euclidean norm [11] will be conducted on features dataset.
   print("printing normalised dataset")
   noramalizedFeatureDF.show()
  
  val Array(trainingDataNew, testDataDFNew) = noramalizedFeatureDF.randomSplit(Array(0.7, 0.3))
  val linearRegressionNewWay = new LinearRegression().
                     setLabelCol("label").
                     setFeaturesCol("normFeatures").
                     setMaxIter(10).
                     setRegParam(0.3).
                     setElasticNetParam(0.8)
 
 // Fit the model
  val lrModel = linearRegressionNewWay.fit(trainingDataNew)

 
  // Print the coefficients and intercept for linear regression
    println(s"Model Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    val trainingSummary = lrModel.summary  
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
 
  //make the prediction
  val newprediction = lrModel.transform(testDataDFNew)
  
  newprediction.show(40)
    
 
}