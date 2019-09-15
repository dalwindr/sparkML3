package com.packt.scalada.learning
import ML_scalaAdvanceMethods._

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator,Normalizer}

import org.apache.log4j._

 
 //vectorizing
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator,Normalizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{ KMeans=> KMeansML}

object Life_expectancyStatisticsDF extends App{
   Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = ML_scalaAdvanceMethods.getSparkSessionConnection("TEst App")
    import spark.sqlContext.implicits._


     val rawDF = spark.read.format("csv").
                     //option("header", "true").
                    //option("inferSchema","true").
                    option("delimiter"," ").
                    load("/Users/keeratjohar2305/Downloads/Dataset/LifeExpentancy.txt").
                    repartition(2).toDF("Counter","y","LifeExp","x","Region").select(col("Counter"),col("LifeExp").cast(DoubleType),col("Region"))
     rawDF.printSchema
     summaryCustomized(rawDF).show()
     
     //Calculating Quantiles in Spark
     
     val Array(min,median,max,tenthPercentile, twentiethPercentile, thirtiethPercentile,eightiethPercentile) = rawDF.stat.approxQuantile("LifeExp",Array(0.0,0.5,1.0,.1,.2,.3,.8),0.0)
     
     println(s""" Calculculting min,max,median using percentile (rawDF.stat.approxQuantile)
                 min= ${min} , 
                 median=${median},
                 max=${max},
                 firstPercentile=${tenthPercentile}, 
                 secondPercentile=${twentiethPercentile}, 
                 thirtiethPercentile=${thirtiethPercentile},
                 eightiethPercentile=${eightiethPercentile}""")
                 
     val DataInArray = rawDF.select("LifeExp").orderBy("LifeExp").collectAsList.toArray
     val min_index_val = (DataInArray.length * 0.0).toInt  
     val median_index_val = (DataInArray.length * .5).toInt 
     val max_index_val = (DataInArray.length -1* 1.0).toInt
     val tenthPercentile_index_val = (DataInArray.length * .1).toInt
     val twentiethPercentile_index_val =  (DataInArray.length * .2).toInt
     val thirtiethPercentile_index_val = (DataInArray.length * .3).toInt
     val eightiethPercentile_index_val = (DataInArray.length * .8).toInt
     
     
          println(s""" Calculculting min,max,median using percentile (sorted Array)
                 min= ${DataInArray(min_index_val)} , 
                 median=${DataInArray(median_index_val)},
                 max=${DataInArray(max_index_val)},
                 firstPercentile=${DataInArray(tenthPercentile_index_val)}, 
                 secondPercentile=${DataInArray(twentiethPercentile_index_val)}, 
                 thirtiethPercentile=${DataInArray(thirtiethPercentile_index_val)},
                 eightiethPercentile=${DataInArray(eightiethPercentile_index_val)}""") 
     
     val Array(xtwentyFifthPercentile,xFifitiethPercentile,xSeventyFifthPercentile) = rawDF.stat.approxQuantile("LifeExp",Array(0.25,0.50,0.75),0.0)
     println(s""" Calculculting 25,50,75  percentiles (rawDF.stat.approxQuantile)
                 xtwentyFifthPercentile= ${xtwentyFifthPercentile} , 
                 xFifitiethPercentile=${xFifitiethPercentile},
                 xSeventyFifthPercentile=${xSeventyFifthPercentile},""")
     
     //distribution = bell curve"

}