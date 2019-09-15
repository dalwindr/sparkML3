
package com.packt.scalada.learning


import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.SparkConf

import org.apache.spark.ml.regression.{LinearRegression,GeneralizedLinearRegression,LinearRegressionModel}
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator,Normalizer}
import org.apache.spark.ml.{Pipeline,Transformer,PipelineModel}
import org.apache.log4j._
import java.lang.System
import ML_scalaAdvanceMethods._
   // import custom function
import org.apache.spark.ml.evaluation.RegressionEvaluator

object housePricePredictionLiRegDF extends App{
 
  Logger.getLogger("org").setLevel(Level.ERROR)
    
     val spark = SparkSession
              .builder()
              .appName("Java Spark SQL basic example")
              .config("spark.master", "local")
              .getOrCreate()
    import spark.sqlContext.implicits._
   
    val mySchema = StructType(Array(
                                    StructField("id",LongType,true),
                                    StructField("date" ,StringType,true),
                                    StructField("price", DoubleType,true),
                                    StructField("bedrooms" ,DoubleType,true),
                                    StructField("bathrooms", DoubleType,true),
                                    StructField("sqft_living" ,DoubleType,true),
                                    StructField("sqft_lot" ,DoubleType,true),
                                    StructField("floors" ,DoubleType,true),
                                    StructField("waterfront" ,DoubleType,true),
                                    StructField("view" ,DoubleType,true),
                                    StructField("grade" ,DoubleType,true),
                                    StructField("sqft_above", DoubleType,true),
                                    StructField("sqft_basement", DoubleType,true),
                                    StructField("yr_built" ,DoubleType,true),
                                    StructField("yr_renovated" ,DoubleType,true),
                                    StructField("zipcode" ,DoubleType,true),
                                    StructField("lat" ,DoubleType,true),
                                    StructField("long" ,DoubleType,true),
                                    StructField("sqft_living15", DoubleType,true),
                                    StructField("sqft_lot15", DoubleType,true)
                                    ))
   
    val rawDFX = spark.read.format("csv").
              option("header","true").
              //option("inferSchema","true").
              schema(mySchema).
              
              //schema(mySchema).
              option("delimiter",",").
              load("/Users/keeratjohar2305/Downloads/ScalaDataAnalysisCookbook/chapter5-learning/kc_house_data.csv").
                    repartition(2)
 
      //Customized Summary Function
 
 
   rawDFX.show()
   rawDFX.printSchema
   summaryCustomized(rawDFX).show()
   
   println("misssing values Filler with mean ..Function invoked")
   // import custom function
   val cleanedDFX = 
     missingValFilled(rawDFX,"sqft_basement").withColumn("label", col("price"))
   summaryCustomized(cleanedDFX).show()
   val Array(traningDFX,testingDFX) = cleanedDFX.randomSplit(Array(0.7,0.3),seed=99999999)


  val featuresCatColNamesX = Seq()
  val categoricalVariablesX = Array("bathrooms","bedrooms","waterfront","view","condition","grade")
  val featuresNumColNamesX = Seq( "sqft_living","sqft_lot","floors","sqft_above","sqft_basement","yr_built","yr_renovated","zipcode","lat","long","sqft_living15","sqft_lot15")
  
  // import custom function
  //import  LinearRegressionWineDF.{FeatureAssembler,CategoricalFeatureVectorzing}
  val featuresAssemblerX = CategoricalFeatureVectorzing(featuresCatColNamesX) ++ FeatureAssembler(featuresCatColNamesX,featuresNumColNamesX )
 
    println(""" 
    //**************************************************
  
  // The Ist way of performing Linear regression
   // Using pipeline 
   
  //*************************************************""")
  
  val lrX = new LinearRegression().setLabelCol("label")
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

  // create pipeline 
  val  pipeline =  new Pipeline().setStages(featuresAssemblerX   ++ Array(lrX))
  
  //traning the model
  val TrainedDFX = pipeline.fit(traningDFX)
 
  
    
   // Model review parameter
   val linRegModelX = TrainedDFX.stages(1).asInstanceOf[LinearRegressionModel]
   val trainingSummary2X = linRegModelX.summary  
 
    
    println(s"Model Coefficients: ${linRegModelX.coefficients} Intercept: ${linRegModelX.intercept}")    
    println(s"numIterations: ${trainingSummary2X.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary2X.objectiveHistory.mkString(",")}]")
    trainingSummary2X.residuals.show()
    println(s"RMSE: ${trainingSummary2X.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary2X.r2}")
   
  //lets make the prediction
  val predictionDFX = TrainedDFX.transform(testingDFX)
  //val featuredTestingDF = pipeline.fit(traningDF).transform(testingDF)
  predictionDFX.show()
  val labelCol="label"
  println ("prediction matrix are : ")
  Seq(("mae" ,"Mean Absalue Error"), ( "mse","mean Square Error"), ("rmse","root Mean Squared Error"), ("r2" , "R2"))
              .foreach{  matrixname=>
                  val regEval = new RegressionEvaluator().
                  setMetricName(matrixname._1).
                  setPredictionCol("prediction").
                  setLabelCol(labelCol)
                  println ("prediction matrix :" + matrixname._2 + " , Values is  " + regEval.evaluate(predictionDFX))
                }
  
  println("""
           //**************************************************************************************************
               II ways of doing  LR \n Here We do considering categorical column ,So pipeline is not required
           //**************************************************************************************************
          """)
  
    // Just to create Dataframe of seleted columns  
  val SelectedFeaturesDF = cleanedDFX.select( (featuresNumColNamesX ++ Seq("label")).map(c => col(c).cast(DoubleType)): _*).cache

  //println("lets prepare feature data for machine learning")
  val vhouse_df11 = FeatureAssembler(featuresCatColNamesX,featuresNumColNamesX )(0).transform(SelectedFeaturesDF).select("features","label")
 
  
  //val Array(traningDFnew,testingDFnew) = vhouse_df.randomSplit(Array(0.7,0.3),seed=99999999)
  vhouse_df11.show()
  //summaryCustomized(vhouse_df11).show()
  
    val noramalizedFeatureDF =  new Normalizer()
                              .setInputCol("features")
                              .setOutputCol("normFeatures")
                              .setP(2.0)
                              .transform(vhouse_df11)
   //Function setP(2.0) [10] has ensured that Euclidean norm [11] will be conducted on features dataset.
   print("printing normalised dataset")
   noramalizedFeatureDF.show()
  
  val Array(trainingDataNew, testDataDFNew) = vhouse_df11.randomSplit(Array(0.7, 0.3))
  val linearRegressionNewWay = new LinearRegression().
                     setLabelCol("label").
                     setFeaturesCol("features").
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
  
  newprediction.show()
   
  
  
}