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

object MultiClassStackOverflow extends App  {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = getSparkSessionConnection("MultiClassDataSetMultiClassAlgoPenBasedDF")
  
    val mySchema = StructType(Array(
                                    StructField("Content",StringType,true),
                                    StructField("label" ,StringType,true)

                                    ))
  
  
   val rawDF = spark.read.format("csv").
              //schema(mySchema).
              option("header","true").
              option("inferSchema","true").
              option("delimiter",",").
              load("/Users/keeratjohar2305/Downloads/Dataset/stack-overflow-data.csv").
                    repartition(6) //.withColumnRenamed("tags","label")
                    //.
                   // withColumnRenamed("quality", "label")
  
  val oldColumnName= rawDF.columns             
  //val newColumnName= rawDF.columns.map(x=> "pix" + x.drop(1))
  //dsShape(rawDF)
  //rawDF.show(false)
  //rawDF.groupBy("tags").count().orderBy(col("Count").asc).show()
  //ML_scalaAdvanceMethods.summaryCustomized(rawDF).show()
  val newDF=rawDF.na.drop()
  
  
  val newDFcleaned = textCleaningDf(newDF,"post")
  println("Returning Cleaned data")
  
   
   
  val labelInderDF = new StringIndexer().setInputCol("tags").setOutputCol("label").fit(newDFcleaned).transform(newDFcleaned).drop("tags").drop("post")
  ML_scalaAdvanceMethods.univariateAnalysis(labelInderDF, Seq("label"))
  labelInderDF.show(10,false)
  
  println(s"catagorical column:\n ${labelInderDF.groupBy(col("label")).count().orderBy().collect()}")
  
  val Array(traningDF,testingDF) = labelInderDF.randomSplit(Array(0.7,0.3),seed=9999)
  println("\n--------------training Data Analysys")
  ML_scalaAdvanceMethods.dsShape(traningDF)
  ML_scalaAdvanceMethods.dataFitnessCheck(traningDF)
  
   
  println("\n--------------training Data Analysys")
  ML_scalaAdvanceMethods.dsShape(testingDF)
  ML_scalaAdvanceMethods.dataFitnessCheck(testingDF)
  

  
  
  //cleaned -> Takens t->  Hashing TF -> TFIDF
  //val tokenizer=new Tokenizer().setInputCol("postCleaned").setOutputCol("tokens")
  //val hashingTf=new HashingTF().setInputCol("tokens").setOutputCol("tf")
  //val idf = new IDF().setInputCol("tf").setOutputCol("tfidf")
  //val featureStringCol = Seq()
  //val featureNumericalCol = traningDF.columns.diff(Array("label")).toSeq
  //val assembler = new VectorAssembler().setInputCols(Array("tfidf")).setOutputCol("features")
  //ML_scalaAdvanceMethods.CallOneVsALLFullAlgo(traningDF,testingDF,"MultiClass",featureStringCol,featureNumericalCol)
  
 import org.apache.spark.ml.feature.StopWordsRemover
 import org.apache.spark.ml.feature.CountVectorizer
 
 val tokenizer=new Tokenizer().setInputCol("postCleaned").setOutputCol("tokens")
 val remover = new StopWordsRemover().setInputCol("tokens").setOutputCol("words")
  // bag of words count
 val countVectors = new CountVectorizer().setInputCol("words").setOutputCol("features").setMinDF(5).setVocabSize(10000)
  
 import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
 val classifier = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)
 val ovr = new OneVsRest().setClassifier(classifier)
  
 import org.apache.spark.ml.Pipeline
 val pipeline=new Pipeline().setStages(Array(tokenizer, remover,countVectors,ovr))
  
 val ovrModel = pipeline.fit(traningDF)

// score the model on test data.
 val predictions = ovrModel.transform(testingDF)
 MultiClassConfusionMatrix(predictions)
  
}