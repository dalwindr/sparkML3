package com.packt.scalada.learning

import java.util.Properties
import org.apache.log4j._
import org.apache.spark.sql.SparkSession
import java.lang.System
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.io.Source
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation
import edu.stanford.nlp.pipeline.Annotation
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.optimization.Updater
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.GeneralizedLinearModel
import org.apache.spark.mllib.regression.GeneralizedLinearAlgorithm
import org.apache.spark.mllib.optimization.SquaredL2Updater
import epic.preprocess.TreebankTokenizer
import epic.preprocess.MLSentenceSegmenter
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.feature.IDFModel
import org.apache.spark.ml.feature.Tokenizer

object BinaryClassificationSpam extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
//  val conf = new SparkConf().setAppName("BinaryClassificationSpam").setMaster("local[2]")
//  val sc = new SparkContext(conf)
//  val sqlContext = new SQLContext(sc)

  //Frankly, we could make this a tuple but this looks neat
  case class Document(label: String, content: String)
  val spark = SparkSession
              .builder()
              .appName("Java Spark SQL basic example")
              .config("spark.master", "local")
              .getOrCreate()
              
  import spark.sqlContext.implicits._    
  
  // SMSSpamCollection file data is tab seperated
  val rowRDD = spark.sparkContext
              .textFile("/Users/keeratjohar2305/Downloads/ScalaDataAnalysisCookbook/chapter5-learning/SMSSpamCollection")
              .map{line =>  val words = line.split("\t"); Document(words.head.trim(), words.tail.mkString(" "))}

  
  //we have two external NLP libraries—the "Stanford" from java world and the second is  "EPIC" libraries from scala.
  
  //1.  Data preparation 
  
  //lets Tokenizing the each document and converting it into LabeledPoints.
  val labeledPointsUsingStanfordNLPRdd=getLabeledPoints(rowRDD, "STANFORD")
  
  //2.  Factoring the Inverse Document Frequency (IDF)
  // TF is term frequency vector which how often a given words occur in a document.
  //IDF, is inverse document frequence dcoument which tell how ofter a given words occurs in the set of documents, 
  //it provide score or weighing factor to the words... provide higher scores to words which are uncomman
  val lpTfIdf=withIdf(labeledPointsUsingStanfordNLPRdd).cache()
  
  //2. Split dataset
  val spamPoints = lpTfIdf.filter(point => point.label == 1).randomSplit(Array(0.8, 0.2))
  val hamPoints = lpTfIdf.filter(point => point.label == 0).randomSplit(Array(0.8, 0.2))

  println("Spam count:" + (spamPoints(0).count) + "::" + (spamPoints(1).count))
  println("Ham count:" + (hamPoints(0).count) + "::" + (hamPoints(1).count))

  val trainingSpamSplit = spamPoints(0)
  val testSpamSplit = spamPoints(1)

  val trainingHamSplit = hamPoints(0)
  val testHamSplit = hamPoints(1)

  val trainingSplit = trainingSpamSplit ++ trainingHamSplit
  val testSplit = testSpamSplit ++ testHamSplit

  val (iterations: Int,stepSize: Int, regParam: Double) = (100, 1, 0.001)
  // 4. get the machine learning algorithm
  val logisticWithSGD = getAlgorithm("LOGSGD", iterations,stepSize,regParam)
  val logisticWithBfgs = getAlgorithm("LOGBFGS", iterations,stepSize,regParam)
  val svmWithSGD = getAlgorithm("SVMSGD", iterations,stepSize,regParam)

 
  //5 create the model and perform prediction
  val logisticWithSGDPredictsActuals=runClassification(logisticWithSGD, trainingSplit, testSplit)
  val logisticWithBfgsPredictsActuals = runClassification(logisticWithBfgs, trainingSplit, testSplit)
  val svmWithSGDPredictsActuals=runClassification(svmWithSGD, trainingSplit, testSplit)

  //Calculate and perform evaluation metrics
  calculateMetrics(logisticWithSGDPredictsActuals, "Logistic Regression with SGD")
  calculateMetrics(logisticWithBfgsPredictsActuals, "Logistic Regression with BFGS")
  calculateMetrics(svmWithSGDPredictsActuals, "SVM with SGD")
  

  def getAlgorithm(algo: String, iterations: Int, stepSize: Double, regParam: Double) = algo match {
    case "LOGSGD" => {
      val algoSGD = new LogisticRegressionWithSGD().setIntercept(true)
      algoSGD.optimizer.setNumIterations(iterations).setStepSize(stepSize).setRegParam(regParam)
      algoSGD
    }
    case "LOGBFGS" => {
      val algoSGD = new LogisticRegressionWithLBFGS().setIntercept(true)
      algoSGD.optimizer.setNumIterations(iterations).setRegParam(regParam)
      algoSGD
    }
    case "SVMSGD" => {
      val algoSGD = new SVMWithSGD().setIntercept(true)
      algoSGD.optimizer.setNumIterations(iterations).setStepSize(stepSize).setRegParam(regParam)
      algoSGD
    }
  }


  /*labeledPointsWithTf.foreach(lp=>{
    println (lp.label +" features : "+lp.features)
  
  })*/

  def withIdf(lPoints: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    // TF = how often a given word occurs in a single document, 
    // HashingTF is vector of documents with map of words with thier occurence
    // IDF =  how often  a given word occures in the entire set of documents and it give highers scores to uncommon words
    // taken out the features
    val hashedFeatures = lPoints.map(lp => lp.features)
    val idf: IDF = new IDF()
    val idfModel: IDFModel = idf.fit(hashedFeatures)

    val tfIdf: RDD[Vector] = idfModel.transform(hashedFeatures)
    
    val lpTfIdf= lPoints.zip(tfIdf).map {
      case (originalLPoint, tfIdfVector) => {
        new LabeledPoint(originalLPoint.label, tfIdfVector)
      }
    }
    
    lpTfIdf
  }
 
  
  /*def withNormalization(lPoints: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    
    val tfIdf: RDD[Vector] = labeledPointsWithTf.map(lp => lp.features)
    
    val normalizer = new Normalizer()
    val lpTfIdfNormalized = labeledPointsWithTf.zip(tfIdf).map {
      case (originalLPoint, tfIdfVector) => {
        new LabeledPoint(originalLPoint.label, normalizer.transform(tfIdfVector))
      }
    }
    lpTfIdfNormalized
  }*/
  
 

  def runClassification(algorithm: GeneralizedLinearAlgorithm[_ <: GeneralizedLinearModel], trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint]): RDD[(Double, Double)] = {
    val model = algorithm.run(trainingData)
    val predicted = model.predict(testData.map(point => point.features))
    val actuals = testData.map(point => point.label)
    val predictsAndActuals: RDD[(Double, Double)] = predicted.zip(actuals)
    predictsAndActuals
  }

  def calculateMetrics(predictsAndActuals: RDD[(Double, Double)], algorithm: String) {

    val accuracy = 1.0 * predictsAndActuals.filter(predActs => predActs._1 == predActs._2).count() / predictsAndActuals.count()
    val binMetrics = new BinaryClassificationMetrics(predictsAndActuals)
    println(s"************** Printing metrics for $algorithm ***************")
    println(s"Area under ROC ${binMetrics.areaUnderROC}")
    //println(s"Accuracy $accuracy")
    
    val metrics = new MulticlassMetrics(predictsAndActuals)
    val f1=metrics.fMeasure
    println(s"F1 $f1")
    
    println(s"Precision : ${metrics.precision}")
    println(s"Confusion Matrix \n${metrics.confusionMatrix}")
    println(s"************** ending metrics for $algorithm *****************")
  }

  def getLabeledPoints(docs: RDD[Document], library: String): RDD[LabeledPoint] = library match {

    case "EPIC" => {

      //Use Scala NLP - Epic
      val labeledPointsUsingEpicRdd: RDD[LabeledPoint] = docs.mapPartitions { docIter =>

        val segmenter = MLSentenceSegmenter.bundled().get
        val tokenizer = new TreebankTokenizer()
        //With HashingTF, we have a map of terms along with their frequency of occurrence in the documents.
        val hashingTf = new HashingTF(5000)

        docIter.map { doc =>
          val sentences = segmenter.apply(doc.content) //splits the paragraph or content into sentences.
          
          // tokenize or split sentences into a map of terms with their frequency of occurrence
          val tokens = sentences.flatMap(sentence => tokenizer(sentence)) 

          //consider only features that are letters or digits and cut off all words that are less than 2 characters
          val filteredTokensDF=tokens.toList.filter(token => token.forall(_.isLetterOrDigit)).filter(_.length() > 1)

          //create each doc into term frequency vector    using      hashingTf
          //we restrict the maximum number of interested terms to 5,000
          // With HashingTF, we have a map of terms along with their frequency of occurrence in the documents.
          val features = hashingTf.transform(filteredTokensDF)
          
          //construct label points to numeric label values and doc which is term frequency vector
          new LabeledPoint(if (doc.label.equals("ham")) 0 else 1,features )
        }
      }.cache()

      labeledPointsUsingEpicRdd

    }

    case "STANFORD" => {
      //we create an NLP pipeline that splits sentences, tokenizes, and  finally reduces the tokens to lemmas:
      def corePipeline(): StanfordCoreNLP = {
        val props = new Properties()
        props.put("annotators", "tokenize, ssplit, pos, lemma")
        new StanfordCoreNLP(props)
      }

      def DocumentLemmatizing(nlpPipeline: StanfordCoreNLP, content: String): List[String] = {
        
        //prepare document  make contents 'annotatable' before we annotate it :-)
        val document = new Annotation(content)
        
        // Annotate the document using nlpPipeline
        nlpPipeline.annotate(document)
        
        //Now Extract all sentences from each document
        val sentences = document.get(classOf[SentencesAnnotation]).asScala

        //Extract lemmas from sentences
        val lemmas = sentences.flatMap { sentence =>
        val tokens = sentence.get(classOf[TokensAnnotation]).asScala
        
        // return lemmatized documents
        tokens.map(token => token.getString(classOf[LemmaAnnotation]))

        }
        //Only lemmas with letters or digits will be considered. Also consider only those words which has a length of at least 2
        lemmas.toList.filter(lemma => lemma.forall(_.isLetterOrDigit)).filter(_.length() > 1)
      }

      val labeledPointsUsingStanfordNLPRdd: RDD[LabeledPoint] = docs.mapPartitions { docIter =>
        val corenlp = corePipeline() // create a  stanford NLP pipeline instance
        val stopwords = Source.fromFile("stopwords.txt").getLines()
        val hashingTf = new HashingTF(5000)  // this will convert doc with tokenized words  into term frequence vector

        docIter.map { doc =>
          
          val lemmas = DocumentLemmatizing(corenlp, doc.content)  
          //remove all the stopwords from the lemma list
          lemmas.filterNot(lemma => lemma.contains(stopwords)) //stopwords.contains(lemma))

          //Generates a term frequency vector from the features
          val features = hashingTf.transform(lemmas)

          //example : List(until, jurong, point, crazy, available, only, in, bugi, great, world, la, buffet, Cine, there, get, amore, wat)
          new LabeledPoint(
            if (doc.label.equals("ham")) 0 else 1,
            features)

        }
      }.cache()

      labeledPointsUsingStanfordNLPRdd
    }

  }
  
}
  