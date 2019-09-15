/*
 * Stochastic Gradient Descent (SGD)
 */
package com.packt.scalada.learning

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.regression.LassoWithSGD
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD
import org.apache.spark.mllib.regression.GeneralizedLinearAlgorithm
import org.apache.spark.mllib.regression.GeneralizedLinearModel
import org.apache.log4j._
import org.apache.spark.sql.SparkSession
import java.lang.System

object LinearRegressionWine extends App {

//  val conf = new SparkConf().setAppName("LinearRegressionWine").setMaster("local[2]")
//  val sc = new SparkContext(conf)
//  val sqlContext = new SQLContext(sc)
  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = SparkSession
              .builder()
              .appName("Java Spark SQL basic example")
              .config("spark.master", "local")
              .getOrCreate()
  import spark.sqlContext.implicits._    
  val rowRDD = spark.sparkContext.textFile("/Users/keeratjohar2305/Downloads/ScalaDataAnalysisCookbook/chapter5-learning/winequality-red.csv").map(line => line.split(";"))

  //Summary stats
  rowRDD.take(10).foreach(x=>println(x.toList))
  //1. Vectors only take double variable ( all the columns value except last should be convert to double) 
  //and considered as feature or input variable or independent vairable or observation
  val featureVector = rowRDD.map(row => Vectors.dense(row.take(row.length-1).map(_.toDouble)))
  
  val stats = Statistics.colStats(featureVector)
  print(s"Max    : ${stats.max}, \nMin    : ${stats.min}, \nMean    : ${stats.mean} and\nVariance : ${stats.variance}")
  
  //2.  last should be convert to double and considered as label point  as output variable or dependent variables
  //The  First parameter to the constructor of the LabeledPoint is the label (y variable), and the second parameter is a vector of input variables.
  val dataPoints = rowRDD.map(row => new LabeledPoint(row.last.toDouble, Vectors.dense(row.take(row.length-1).map(str => str.toDouble)))).cache()
  //System.exit(1)
  
  //split dataset into traing dataset and test dataset
  val dataSplit = dataPoints.randomSplit(Array(0.8, 0.2))
  val trainingSet = dataSplit(0)
  val testSet = dataSplit(1)
  
  // 3. Preparing DataSet for ML

  //3.1 Let's create scaler from  the training dataset 
  //It is always recommended that the input variables (X) or features have a mean of 0. 
  //This is easily achieved with the help of the StandardScaler built into the Spark ML library itself.
  val scaler = new StandardScaler(withMean = true, withStd = true).fit(trainingSet.map(dp => dp.features))
  
  //3.2 Use the training scaler to scala traning dataset well as test dataset
  val scaledTrainingSet = trainingSet.map(dp => new LabeledPoint(dp.label, scaler.transform(dp.features))).cache()
  
   //3.2  Use the training scaler to test dataset also
  val scaledTestSet = testSet.map(dp => new LabeledPoint(dp.label, scaler.transform(dp.features))).cache()

  // 4 . SGD has also two other more optimised and simple algo lasso and ridghe
  //  which can overcome the problem of overfitting the model and these two are regularized model
  // make prediction accuratly
  
 
 //  parameters are required  to create model using LinearRegressionWithSGD  1. setIntercept, 2. optimizer.setNumIterations, 3.setStepSize
  val iterations = 1000
  val stepSize = 1
  
  
  //4.1 Deciding ml algorithm and its parameters 
  val linearRegWithoutRegularization=algorithm("linear", iterations, stepSize)
  
  //4.2 Create models from the algo using training dataset and perform prediction on test dataset
  val linRegressionPredictActuals = runRegression(linearRegWithoutRegularization)

  // 
  
  val lasso=algorithm("lasso", iterations, stepSize)
  val lassoPredictActuals = runRegression(lasso)

  val ridge=algorithm("ridge", iterations, stepSize)
  val ridgePredictActuals = runRegression(ridge)

  //calculate and perform evaluation metrics 
  calculateMetrics(linRegressionPredictActuals, "Linear Regression with SGD")
  calculateMetrics(lassoPredictActuals, "Lasso Regression with SGD")
  calculateMetrics(ridgePredictActuals, "Ridge Regression with SGD")
  
  
  def algorithm(algo: String, iterations: Int, stepSize: Double) = algo match {
    case "linear" => {
      val algoSGD = new LinearRegressionWithSGD().setIntercept(true)
      algoSGD.optimizer.setNumIterations(iterations).setStepSize(stepSize).setMiniBatchFraction(0.05)
      algoSGD
    }
    case "lasso" => {
      val algoSGD = new LassoWithSGD().setIntercept(true)
      algoSGD.optimizer.setNumIterations(iterations).setStepSize(stepSize).setRegParam(0.001).setMiniBatchFraction(0.05)
      algoSGD
    }
    case "ridge" => {
      val algoSGD = new RidgeRegressionWithSGD().setIntercept(true)
      algoSGD.optimizer.setNumIterations(iterations).setStepSize(stepSize).setRegParam(0.001).setMiniBatchFraction(0.05)
      algoSGD
    }
  }

  def runRegression(algorithm: GeneralizedLinearAlgorithm[_ <: GeneralizedLinearModel]):RDD[(Double,Double)] = {
    val model = algorithm.run(scaledTrainingSet) //Let's pass in the training split 

    val predictions: RDD[Double] = model.predict(scaledTestSet.map(point => point.features))
    val actuals: RDD[Double] = scaledTestSet.map(point => point.label)

    //Let's go ahead and calculate the Residual Sum of squares
    val predictsAndActuals: RDD[(Double, Double)] = predictions.zip(actuals)
    predictsAndActuals
  }

  def calculateMetrics(predictsAndActuals: RDD[(Double, Double)], algorithm: String) {

   
    
    val sumSquaredErrors = predictsAndActuals.map {
      case (pred, act) =>
        math.pow(act - pred, 2)
    }.sum()

    val meanSquaredError = sumSquaredErrors / scaledTestSet.count

   println(s"\n************** Printing metrics for $algorithm *****************")

    println(s"Matrix 1. Sum of Squared Error is $sumSquaredErrors")
    println(s"Matrix 2. Mean Squared Error is $meanSquaredError")
    
    val meanQuality = dataPoints.map(point => point.label).mean()
    val totalSumOfSquares = predictsAndActuals.map {
      case (pred, act) =>
        math.pow(act - meanQuality, 2)
    }.sum()

    println(s"Matrix 3. Sum of Square Total is $totalSumOfSquares")
    val rss = 1 - sumSquaredErrors / totalSumOfSquares

    println(s"Matrix 4. Residual sum of squares is $rss")
    

  }
  /*
   * 
   * With supervised learning, in order for the algorithm to learn the relationship between the input and the output features,
   * we provide a set of manually curated values for the target variable (y - label points) against a set of input variables (x - the features) . 
   * 
   * We call it the training set. The learning algorithm then has to go over our training set, perform some optimization, and come up with a model 
   * that has the least cost—deviation from the true values. 
   * 
   * So technically, we have two algorithms for every learning problem: 
   *       An algorithm that comes up with the function and (an initial set of) weights for each of the x features, and 
   *       A supporting algorithm (also called cost minimization or optimization algorithm) that looks at our function parameters (feature weights) and 
   *       tries to minimize the cost as much as possible.
   *       
   *  There are a variety of cost minimization algorithms, but one of the most popular is gradient descent. 
   *  Imagine gradient descent as climbing down a mountain. The height of the mountain represents the cost, and the plain represents the feature weights. 
   *  The highest point is your function with the maximum cost, and the lowest point has the least cost. Therefore, our intention is to walk down
   *   the mountain. 
   *   
   *   What gradient descent does is as follows: for every single step down the slope that it takes of a particular size (the step size), it goes through 
   *   the entire dataset (!) and updates all the values of the weights for x features. This goes on until it reaches a state where the cost is the minimum. 
   *   This  avor of gradient descent, in which it sees all of the data per iteration and updates all the parameters during every iteration, 
   *   is called batch gradient descent. 
   *   
   *   The trouble with using this algorithm against the size of the data that Spark aims to handle is that going 
   *   through millions of rows per iteration is definitely not optimal. So, Spark uses a variant of gradient descent, called Stochastic Gradient Descent (SGD), 
   *   wherein the parameters are updated for each training example as it looks at it one by one. In this way, it starts making progress almost immediately, 
   *   and therefore the computational effort is considerably reduced. The SGD settings can be customized using the optimizer attribute inside each of the ML
   *    algorithm. We'll look at this in detail in the recipes.
   * 
   *  Model creation using  LinearRegressionWithSGD
   *  This just involves creating an instance of LinearRegressionWithSGD and passing in a three parameters: 
   *  				  one for the LinearRegression algorithm and 
   *   				two for the SGD. 
   *   
   *   The SGD parameters can be accessed through the use of the optimizer attribute inside LinearRegressionWithSGD:
   *   
     parameter 1:  setIntercept:
     					 While predicting, we are more interested in the slope. This setting will force the algorithm to  find the intercept too.
     					 
     parameter 2:  optimizer.setNumIterations: 
     					This determines the number of iterations that our algorithm needs to go through on the training set before  
     finalizing the hypothesis/predictions. An optimal number would be 10^6 divided by the number of instances in your dataset. 
     In our case, we'll set it to 1000.

     parameter 3 : setStepSize: 
      					This tells the gradient descent algorithm while it tries to reduce the parameters how big a step it needs to take 
      during every iteration. Setting this parameter is really tricky because we would like the SGD to take bigger steps in the beginning and smaller
      steps towards the convergence. Setting a  xed small number would slow down the algorithm, and setting a  xed bigger number would not give us a 
      function that is a reasonable minimum. The way Spark handles our setStepSize input parameter is as follows: it divides the input parameter by a 
      root of the iteration number. So initially, our step size is huge, and as we go further down, it becomes smaller and smaller. The default step size 
      parameter is 1.
   * 
   * 
   */
  
  

}