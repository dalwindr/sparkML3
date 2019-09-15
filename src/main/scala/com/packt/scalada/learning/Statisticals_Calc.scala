package com.packt.scalada.learning

import org.apache.log4j._
import ML_scalaAdvanceMethods._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._

object Statisticals_Calc extends App {
 
     Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = ML_scalaAdvanceMethods.getSparkSessionConnection("TEst App2")
    import spark.sqlContext.implicits._
    println("""
       6 important functions, including:

              1)  Random data generation
              2)  Summary and descriptive statistics
              3)  Sample covariance and correlation
              4)  Cross tabulation (a.k.a. contingency table)
              5)  Frequent items
              6)  Mathematical functions
      """)
  
    val df1 = spark.range(10).toDF()
    df1.show()   
    
    //# 1. Generate two other columns using uniform distribution and normal distribution.
    val df = df1.select(col("id"), rand(seed=10).alias("uniform"), randn(seed=27).alias("normal"))
    df.show() 
    
    
    // # A slightly different way to generate the two random columns
    val df2 = df1.withColumn("uniform", rand(seed=10)).withColumn("normal", randn(seed=27)).withColumn("uniform1", rand(seed=30))
    df2.show() 
    
    
    //# 2. you can also run describe on all or selected columns:
    ML_scalaAdvanceMethods.summaryCustomized(df2).show()
    
    //control the list of descriptive statistics and the columns they apply to using the normal select on a DataFrame:
    df2.agg(mean("uniform"), min("uniform"), max("uniform")).show()
    
    
    // # 3. Sample covariance and correlation
   println(s"""covariance =   
                         ${df2.stat.cov("uniform", "uniform1")}
                         ${df2.stat.cov("id", "id") } """)
   
   println(s"""Correlation =    
                             ${df2.stat.cov("uniform", "uniform1")}
                             ${df2.stat.cov("id", "id") } """)
   
   
                             
   //# Create a DataFrame with two columns (name, item)
   val names = Array("Alice", "Bob", "Mike")
   val items = Array("milk", "bread", "butter", "apples", "oranges")
   val namesItemSeq = for ( i <- 0 to 100) yield (names(i % 3), items(i % 5))
   
   //# 4. Cross Tab
   //One important thing to keep in mind is that the cardinality of columns we run crosstab on cannot be too big. That is to say, 
   //the number of distinct “name” and “item” cannot be too large. Just imagine if “item” contains 1 billion distinct entries: how would you 
   //fit that table on your screen?!
   val namesItemSeqDF = spark.sqlContext.createDataFrame(namesItemSeq).toDF("name", "item")
   namesItemSeqDF.stat.crosstab("name", "item").show()

   
   
   //# Create a DataFrame with two columns (name, item)
   val FewNumberSeq = for (i<- 0 to 100) yield if ( i % 2   == 0 ) (1, 2, 3) else (i, 2 * i, i % 4) 
   val FewNumberSeqDF = spark.sqlContext.createDataFrame(FewNumberSeq).toDF("a", "b", "c")
   
   //5. Frequent Items
   //Figuring out which items are frequent in each column can be very useful to understand a dataset. 
   //In Spark 1.4, users will be able to find the frequent items for a set of columns using DataFrames. We have implemented an one-pass 
   //algorithm proposed by Karp et al. This is a fast, approximate algorithm that always return all the frequent items that appear in a
   //user-specified minimum proportion of rows. Note that the result might contain false positives, i.e. items that are not frequent.
   FewNumberSeqDF.stat.freqItems(Array("a", "b", "c"), 0.4).show()
   
   val mySeq= Seq(
       ("Alice","Math",21),
       ("Bob", "CS",23),
       ("Carl","Math1",25),
       ("Carl1","CS",25),
       ("Carl2","CS2",25),
       ("Carl3","CS2",25),
       ("Carl4","CS2",25),
       ("Carl5","CS",25),
       ("Carl6","CS1",25),
       ("Carl7","CS1",25),
       ("Carl8","CS1",25)        
   )
   
     val testDF = spark.sqlContext.createDataFrame(mySeq).toDF("name", "dept", "age")
     testDF.stat.freqItems(Seq("dept"),.3).show()
     
     
     df.select(col( "uniform"),toDegrees("uniform"),(pow(cos(col("uniform")), 2) + pow(sin(col("uniform")), 2)).alias("cos^2 + sin^2")).show()
   
}