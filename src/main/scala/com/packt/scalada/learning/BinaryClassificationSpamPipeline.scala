package com.packt.scalada.learning

import java.lang.System
import org.apache.log4j._
import org.apache.spark.SparkConf
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator}
import org.apache.spark.ml.{Pipeline,Transformer,PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassifier,RandomForestClassificationModel,LogisticRegression,LogisticRegressionModel}
import org.apache.spark.ml.linalg.{Vectors,DenseVector}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics,MulticlassMetrics}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._

// text analsysys
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.feature.IDF


object BinaryClassificationSpamPipeline extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  
  val spark = SparkSession
              .builder()
              .appName("Java Spark SQL basic example")
              .config("spark.master", "local")
              .getOrCreate()
 
   import spark.sqlContext.implicits._ 
  
   val myFunc=( str: String) => if ( str =="spam" ) 1.0 else 0.0 
   val funcUDF = udf(myFunc)
   spark.udf.register("funcUDF",funcUDF)
   
   val myschema= StructType(Seq(
       StructField("myLabel",StringType,true),
       StructField("content",StringType,true)))
 
    val rawDF = spark.read.format("csv")
                      .schema(myschema)
                      .option("delimiter","\t")
                      .load("SMSSpamCollection")
                      .withColumn("label",funcUDF(col("myLabel")) )
  
               
  rawDF.show(false)
  rawDF.printSchema
  rawDF.columns
  rawDF.summary().show()
  
  

   val Array(spamTrainingDF,spamtestDF) = rawDF.where(col("label")===1).randomSplit(Array(0.8, 0.2),seed = 12345)  
   val Array(hamTrainingDF,hamtestDF) = rawDF.filter(col("label")===0).randomSplit(Array(0.8, 0.2))
  
   val trainingDFrame = spamTrainingDF.union(hamTrainingDF)
   val testDFrame = spamtestDF.union(hamtestDF)

  val tokenizer=new Tokenizer().setInputCol("content").setOutputCol("tokens")
  val hashingTf=new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("tf")
  val idf = new IDF().setInputCol(hashingTf.getOutputCol).setOutputCol("tfidf")
  val assembler = new VectorAssembler().setInputCols(Array("tfidf", "label")).setOutputCol("features")
 
  val logisticRegression=new LogisticRegression().setFeaturesCol("features").setLabelCol("label").setMaxIter(10)
  
  //1 create pipeline
  val pipeline=new Pipeline().setStages(Array(tokenizer, hashingTf, idf, assembler,logisticRegression))
  
  
  //2. create model
  val model=pipeline.fit(trainingDFrame)
  
  
  //3. perform perdiction
  val prediction = model.transform(testDFrame) 
  prediction.show(false)
   // prediction.select("label","content","probability","prediction").show(false)
  
  //4 . perform calculations
  calculateMetrics(prediction, "Without Cross validation")
  
  //5. Using Cross validator
  //This will provide the cross validator various parameters to choose from 	
  val paramGrid=new ParamGridBuilder()
  	.addGrid(hashingTf.numFeatures, Array(1000, 5000, 10000))
  	.addGrid(logisticRegression.regParam, Array(1, 0.1, 0.03, 0.01))
  	.build()
  	
  val crossValidator=new CrossValidator()
  	.setEstimator(pipeline)
  	.setEvaluator(new BinaryClassificationEvaluator())
  	.setEstimatorParamMaps(paramGrid)
  	.setNumFolds(10)

  val bestModel=crossValidator.fit(trainingDFrame)
  val cvDF= bestModel.transform(testDFrame)  
  calculateMetrics(cvDF, "Cross validation")


  def calculateMetrics(predictsAndActualsNoCV: org.apache.spark.sql.DataFrame, algorithm: String) {

    println(s"************** Printing metrics for $algorithm ***************")   
      // Lets start calculatring confusion matrix
    val lp = predictsAndActualsNoCV.select( "label", "prediction").createOrReplaceTempView("prediction_view")
    var counttotal = spark.sql("select count(1) from prediction_view").collect()(0).mkString
    val correct = spark.sql("select count(1) from prediction_view where label == prediction").collect()(0).mkString
    val wrong = spark.sql("select count(1) from prediction_view where label != prediction").collect()(0).mkString
    
    val truep = spark.sql("select count(1) from prediction_view where label == prediction and prediction = 1.0 and label=1.0").collect()(0).mkString
    val falseP = spark.sql("select count(1) from prediction_view where label != prediction and prediction = 1.0 and label=0.0").collect()(0).mkString
    val falseN = spark.sql("select count(1) from prediction_view where label == prediction and prediction = 0.0 and label=0.0").collect()(0).mkString
    val trueN = spark.sql("select count(1) from prediction_view where label != prediction and prediction = 0.0 and label=1.0").collect()(0).mkString
    
    print("\ncounttotal:", counttotal)
    print("\ncorrect:",correct)
    print("\nwrong:",wrong)
    
    print("\ntruep:",truep)
    print("\nfalseP:",falseP)
    print("\nfalseN:",falseN)
    print("\ntrueN:",trueN)
    
    
    val ratioWrong=wrong.toDouble/counttotal.toDouble
    val ratioCorrect=correct.toDouble/counttotal.toDouble
    
    println("ratioWrong =", ratioWrong)
    println("ratioCorrect =", ratioCorrect)
    
   
    // There are two of calculating precision - recall (PR) and reciever operating characterstics
    //Using rdd
    val  predictionAndLabels =predictsAndActualsNoCV.select("rawPrediction", "label").rdd.map(x=>  (x(0).asInstanceOf[org.apache.spark.ml.linalg.DenseVector](1) , x(1).asInstanceOf[Double]))
    //predictionAndLabels.collect().foreach(println)
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    println("\narea under the precision-recall curve (areaUnderPR): " + metrics.areaUnderPR)
    println("\narea under the receiver operating characteristic (areaUnderROC) curve : " + metrics.areaUnderROC)
  
    
    // Using DataFrame itself
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
    val accuracy = evaluator.evaluate(predictsAndActualsNoCV)
    print(s"\nAccuracy  Per BinaryClassificationEvaluator(--areaUnderROC): $accuracy")
    
    // Using DataFrame itself
    val evaluator1 = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderPR")
    val accuracy1 = evaluator.evaluate(predictsAndActualsNoCV)
    print(s"\nAccuracy  Per BinaryClassificationEvaluator (--areaUnderPR): $accuracy1")
    
    val predictsAndActuals =  predictsAndActualsNoCV.rdd.map(r => (r.getAs[Double]("label"), r.getAs[Double]("prediction")))
 
    val accuracy11 = predictsAndActuals.filter(predActs => predActs._1 == predActs._2).count() // predictsAndActuals.count()
    println(s"Accuracy $accuracy11")

    val metrics11 = new MulticlassMetrics(predictsAndActuals)
    println(s"metrics11  MulticlassMetrics : ${metrics11.accuracy}, \nmetrics11  MulticlassMetric ${metrics11.precision},\nmetrics11  MulticlassMetric ${metrics11.recall}")
    println(s"Confusion Matrix \n${metrics11.confusionMatrix}")
    println(s"************** ending metrics for $algorithm *****************")

   
   }

  

}
  