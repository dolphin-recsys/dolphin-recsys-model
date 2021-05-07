package com.sparrowrecsys.offline.spark.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
//import org.apache.spark.sql.functions.{format_number, _}
import org.apache.spark.sql.types.{DecimalType, FloatType, IntegerType, LongType}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import redis.clients.jedis.Jedis
import redis.clients.jedis.params.SetParams
import scala.collection.{JavaConversions, mutable}

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import scala.util.control.Breaks.{break, breakable}

object FeatureEngForourdata {

  val NUMBER_PRECISION = 2
  val redisEndpoint = "localhost"
  val redisPort = 6379
  val secretKey = 123456

  def addSampleLabel(ratingSamples:DataFrame): DataFrame ={
    ratingSamples.show(10, truncate = false)
    ratingSamples.printSchema()
    val sampleCount = ratingSamples.count()
    println("debug 000")
    ratingSamples.groupBy(col("评分")).count().orderBy(col("评分"))
      .withColumn("percentage", col("count")/sampleCount).show(10,truncate = false)
    ratingSamples.show(10)
    ratingSamples.withColumn("label", when(col("评分") >= 6, 1).otherwise(0))
  }

  def addMovieFeatures(movieSamples:DataFrame, ratingSamples:DataFrame): DataFrame ={

    // like the function of merge
    //add movie basic features
    val samplesWithMovies1 = ratingSamples.join(movieSamples, Seq("movieid","类型","电影名"), "left")

    //split direcors and actors
    val samplesWithMovies2 = samplesWithMovies1.withColumn("movieactor1",split(col("主演"),"\\|").getItem(0))
      .withColumn("movieactor2",split(col("主演"),"\\|").getItem(1))
      .withColumn("movieactor3",split(col("主演"),"\\|").getItem(2))
      .withColumn("movieactor4",split(col("主演"),"\\|").getItem(3))

    println("debug 002")
    samplesWithMovies2.show(5, truncate = false)
    val samplesWithMovies3 = samplesWithMovies2.withColumn("moviedirecotr1",split(col("导演"),"\\|").getItem(0))
      .withColumn("moviedirecotr2",split(col("主演"),"\\|").getItem(1))


    println("debug 003")
    samplesWithMovies3.show(5, truncate = false)
    //add rating features
    val movieRatingFeatures = samplesWithMovies3.groupBy(col("movieid"))
      .agg(count(lit(1)).as("movieRatingCount"),
        format_number(avg(col("评分")), NUMBER_PRECISION).as("movieAvgRating"),
        stddev(col("评分")).as("movieRatingStddev"))
    .na.fill(0).withColumn("movieRatingStddev",format_number(col("movieRatingStddev"), NUMBER_PRECISION))
    println("debug 004")
    movieRatingFeatures.show(5, truncate = false)

    //join movie rating features
    val samplesWithMovies4 = samplesWithMovies3.join(movieRatingFeatures, Seq("movieid"), "left")
    samplesWithMovies4.printSchema()
    println("debug 005")
    samplesWithMovies4.show(10, truncate = false)



    samplesWithMovies4
  }

  def addUserFeatures(ratingSamples:DataFrame): DataFrame ={
    val samplesWithUserFeatures = ratingSamples
      .withColumn("userPositiveHistory", collect_list(when(col("label") === 1, col("movieid")).otherwise(lit(null)))
        .over(Window.partitionBy("用户ID")
          .orderBy(col("评论时间")).rowsBetween(-100, -1)))
      .withColumn("userPositiveHistory", reverse(col("userPositiveHistory")))
      .withColumn("userRatedMovie1",col("userPositiveHistory").getItem(0))
      .withColumn("userRatedMovie2",col("userPositiveHistory").getItem(1))
      .withColumn("userRatedMovie3",col("userPositiveHistory").getItem(2))
      .withColumn("userRatedMovie4",col("userPositiveHistory").getItem(3))
      .withColumn("userRatedMovie5",col("userPositiveHistory").getItem(4))
      .withColumn("userPositiveType", collect_list(when(col("label") === 1, col("类型")).otherwise(lit(null)))
        .over(Window.partitionBy("用户ID")
          .orderBy(col("评论时间")).rowsBetween(-100, -1)))
      .withColumn("userPositiveType", reverse(col("userPositiveType")))
      .withColumn("userPositiveType1",col("userPositiveType").getItem(0))
      .withColumn("userPositiveType2",col("userPositiveType").getItem(1))
      .withColumn("userPositiveType3",col("userPositiveType").getItem(2))
      .withColumn("userPositiveType4",col("userPositiveType").getItem(3))
      .withColumn("userPositiveType5",col("userPositiveType").getItem(4))
      .withColumn("userPositivearea", collect_list(when(col("label") === 1, col("地区")).otherwise(lit(null)))
      .over(Window.partitionBy("用户ID")
        .orderBy(col("评论时间")).rowsBetween(-100, -1)))
      .withColumn("userPositivearea", reverse(col("userPositivearea")))
      .withColumn("userPositivearea1",col("userPositivearea").getItem(0))
      .withColumn("userPositivearea2",col("userPositivearea").getItem(1))
      .withColumn("userPositivearea3",col("userPositivearea").getItem(2))
      .withColumn("userPositivearea4",col("userPositivearea").getItem(3))
      .withColumn("userPositivearea5",col("userPositivearea").getItem(4))
      .withColumn("userPositivefeature", collect_list(when(col("label") === 1, col("特色")).otherwise(lit(null)))
        .over(Window.partitionBy("用户ID")
          .orderBy(col("评论时间")).rowsBetween(-100, -1)))
      .withColumn("userPositivefeature", reverse(col("userPositivefeature")))
      .withColumn("userPositivefeature1",col("userPositivefeature").getItem(0))
      .withColumn("userPositivefeature2",col("userPositivefeature").getItem(1))
      .withColumn("userPositivefeature3",col("userPositivefeature").getItem(2))
      .withColumn("userPositivefeature4",col("userPositivefeature").getItem(3))
      .withColumn("userPositivefeature5",col("userPositivefeature").getItem(4))
      .withColumn("userRatingCount", count(lit(1))
        .over(Window.partitionBy("用户ID")
          .orderBy(col("评论时间")).rowsBetween(-100, -1)))
      .withColumn("userAvgRating", format_number(avg(col("评分"))
        .over(Window.partitionBy("用户ID")
          .orderBy(col("评论时间")).rowsBetween(-100, -1)), NUMBER_PRECISION))
      .withColumn("userRatingStddev", stddev(col("评分"))
        .over(Window.partitionBy("用户ID")
          .orderBy(col("评论时间")).rowsBetween(-100, -1)))
      .na.fill(0)
      .withColumn("userRatingStddev",format_number(col("userRatingStddev"), NUMBER_PRECISION))
      .drop("主演", "地区", "导演","特色","类型","userPositiveHistory","userPositiveType","userPositivearea","userPositivefeature")
      .filter(col("userRatingCount") > 1)

    samplesWithUserFeatures.printSchema()
    samplesWithUserFeatures.show(30, truncate = false)

      samplesWithUserFeatures
  }

  def extractAndSaveMovieFeaturesToRedis(samples:DataFrame): DataFrame = {
    val movieLatestSamples = samples.withColumn("movieRowNum", row_number()
      .over(Window.partitionBy("movieid")
        .orderBy(col("评论时间").desc)))
      .filter(col("movieRowNum") === 1)
      .select("movieid", "movieactor1","movieactor2","movieactor3","movieactor4",
        "moviedirecotr1","moviedirecotr2","movieAvgRating", "movieRatingStddev","movieRatingCount")
      .na.fill("")

    movieLatestSamples.printSchema()
    movieLatestSamples.show(20, truncate = false)

    val movieFeaturePrefix = "mf:"

    val redisClient = new Jedis(redisEndpoint, redisPort)
    val params = SetParams.setParams()
    //set ttl to 24hs * 30
    params.ex(60 * 60 * 24 * 30)
    val sampleArray = movieLatestSamples.collect()
    println("total movie size:" + sampleArray.length)
    var insertedMovieNumber = 0
    val movieCount = sampleArray.length
    println("movieCount is")
    println(movieCount)
    for (sample <- sampleArray){
      println("sample is")
      println(sample)
      val movieKey = movieFeaturePrefix + sample.getAs[String]("movieid")
      val valueMap = mutable.Map[String, String]()
      valueMap("movieactor1") = sample.getAs[String]("movieactor1")
      valueMap("movieactor2") = sample.getAs[String]("movieactor2")
      valueMap("movieactor3") = sample.getAs[String]("movieactor3")
      valueMap("movieactor4") = sample.getAs[String]("movieactor4")
      valueMap("moviedirecotr1") = sample.getAs[String]("moviedirecotr1")
      valueMap("moviedirecotr2") = sample.getAs[String]("moviedirecotr2")
      valueMap("movieRatingCount") = sample.getAs[Long]("movieRatingCount").toString
      valueMap("movieAvgRating") = sample.getAs[String]("movieAvgRating")
      valueMap("movieRatingStddev") = sample.getAs[String]("movieRatingStddev")

      redisClient.hset(movieKey, JavaConversions.mapAsJavaMap(valueMap))
      insertedMovieNumber += 1
      if (insertedMovieNumber % 100 ==0){
        println(insertedMovieNumber + "/" + movieCount + "...")
      }
    }

    redisClient.close()
    movieLatestSamples
  }

  def splitAndSaveTrainingTestSamples(samples:DataFrame, savePath:String)={
    //generate a smaller sample set for demo
    val smallSamples = samples.sample(0.1)
    val allsmallSamples = samples
    //split training and test set by 8:2
    val Array(training, test) = smallSamples.randomSplit(Array(0.8, 0.2))

    val sampleResourcesPath = this.getClass.getResource(savePath)
    println("type training")
    println(training.getClass)
    println("training count")
    println( training.count())
    println("test count")
    println( test.count())
    println("all count")
    println( allsmallSamples.count())
    training.coalesce(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath+"/trainingSamples")
    test.coalesce(1).repartition(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath+"/testSamples")
    allsmallSamples.coalesce(1).repartition(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath+"/allSamples")
  }

  def splitAndSaveTrainingTestSamplesByTimeStamp(samples:DataFrame, savePath:String)={
    //generate a smaller sample set for demo
    val smallSamples = samples.sample(0.1).withColumn("timestampLong", col("timestamp").cast(LongType))

    val quantile = smallSamples.stat.approxQuantile("timestampLong", Array(0.8), 0.05)
    val splitTimestamp = quantile.apply(0)

    val training = smallSamples.where(col("timestampLong") <= splitTimestamp).drop("timestampLong")
    val test = smallSamples.where(col("timestampLong") > splitTimestamp).drop("timestampLong")

    val sampleResourcesPath = this.getClass.getResource(savePath)
    //training.coalesce(1).write.option("header", "true").csv("sample_file.csv")
    println("training count")
    println( training.count())
    println("test count")
    println( test.count())
    training.coalesce(1).write.option("header", "true")
      .csv(sampleResourcesPath+"trainingSamples.csv")
    test.coalesce(1).repartition(1).write.option("header", "true")
      .csv(sampleResourcesPath+"testSamples.csv")
  }


  def extractAndSaveUserFeaturesToRedis(samples:DataFrame): DataFrame = {
    val userLatestSamples = samples.withColumn("userRowNum", row_number()
      .over(Window.partitionBy("用户ID")
        .orderBy(col("评论时间").desc)))
      .filter(col("userRowNum") === 1)
      .select("用户ID","userRatedMovie1", "userRatedMovie2","userRatedMovie3","userRatedMovie4","userRatedMovie5",
        "userPositiveType1","userPositiveType2","userPositiveType3","userPositiveType4","userPositiveType5",
        "userPositivearea1","userPositivearea2","userPositivearea3","userPositivearea4","userPositivearea5",
        "userPositivefeature1","userPositivefeature2","userPositivefeature3","userPositivefeature4","userPositivefeature5",
        "userRatingCount","userAvgRating", "userRatingStddev")
      .na.fill("")

    userLatestSamples.printSchema()
    userLatestSamples.show(20, truncate = false)

    val userFeaturePrefix = "uf:"

    val redisClient = new Jedis(redisEndpoint, redisPort)
    val params = SetParams.setParams()
    //set ttl to 24hs * 30
    params.ex(60 * 60 * 24 * 30)
    val sampleArray = userLatestSamples.collect()
    println("total user size:" + sampleArray.length)
    var insertedUserNumber = 0
    val userCount = sampleArray.length
    for (sample <- sampleArray){
      val userKey = userFeaturePrefix + sample.getAs[String]("用户ID")
      val valueMap = mutable.Map[String, String]()
      valueMap("userRatedMovie1") = sample.getAs[String]("userRatedMovie1")
      valueMap("userRatedMovie2") = sample.getAs[String]("userRatedMovie2")
      valueMap("userRatedMovie3") = sample.getAs[String]("userRatedMovie3")
      valueMap("userRatedMovie4") = sample.getAs[String]("userRatedMovie4")
      valueMap("userRatedMovie5") = sample.getAs[String]("userRatedMovie5")
      valueMap("userPositiveType1") = sample.getAs[String]("userPositiveType1")
      valueMap("userPositiveType2") = sample.getAs[String]("userPositiveType2")
      valueMap("userPositiveType3") = sample.getAs[String]("userPositiveType3")
      valueMap("userPositiveType4") = sample.getAs[String]("userPositiveType4")
      valueMap("userPositiveType5") = sample.getAs[String]("userPositiveType5")
      valueMap("userPositivearea1") = sample.getAs[String]("userPositivearea1")
      valueMap("userPositivearea2") = sample.getAs[String]("userPositivearea2")
      valueMap("userPositivearea3") = sample.getAs[String]("userPositivearea3")
      valueMap("userPositivearea4") = sample.getAs[String]("userPositivearea4")
      valueMap("userPositivearea5") = sample.getAs[String]("userPositivearea5")
      valueMap("userPositivefeature1") = sample.getAs[String]("userPositivefeature1")
      valueMap("userPositivefeature2") = sample.getAs[String]("userPositivefeature2")
      valueMap("userPositivefeature3") = sample.getAs[String]("userPositivefeature3")
      valueMap("userPositivefeature4") = sample.getAs[String]("userPositivefeature4")
      valueMap("userPositivefeature5") = sample.getAs[String]("userPositivefeature5")
      valueMap("userRatingCount") = sample.getAs[Long]("userRatingCount").toString
      valueMap("userAvgRating") = sample.getAs[String]("userAvgRating")
      valueMap("userRatingStddev") = sample.getAs[String]("userRatingStddev")

      redisClient.hset(userKey, JavaConversions.mapAsJavaMap(valueMap))
      insertedUserNumber += 1
      if (insertedUserNumber % 100 ==0){
        println(insertedUserNumber + "/" + userCount + "...")
      }
    }

    redisClient.close()
    userLatestSamples
  }



  def main(args: Array[String]): Unit = {
    //    val path="D:/python/SparrowRecSys/target/classes/webroot/test0427/"
    //    val file=new File(path)
    //    if(file.exists()) {
    //      println("存在!")
    //    }else{
    //      println("不存在!")
    //    }
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("featureEngineering")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()

    val movieResourcesPath = this.getClass.getResource("/webroot/ourdatas/movie.csv")

    val movieSamples = spark.read.format("csv").option("header", "true").load(movieResourcesPath.getPath)
      .withColumnRenamed("评分", "豆瓣网评分")
    //println("some try to change the movie datas")
    //val df2 = movieSamples.withColumnRenamed("电影名","moviename")
    //df2.createOrReplaceTempView("moviedata")
    //val distinct_mac_DF = spark.sql("SELECT  DISTINCT moviename FROM moviedata")
    //val moviecount = spark.sql("SELECT DISTINCT  moviename FROM moviedata").count().toInt
    //val movie_names = distinct_mac_DF.collect.toList.map(_.toString())
    //var movienametoindex:Map[String,Int] = Map()
    //for(i <- movie_names) {
    //  println(i + "\t")
    //  println(movie_names.indexOf(i) )
    //  movienametoindex += (i -> movie_names.indexOf(i))
    //}
    //println("打印映射")
    //println(movienametoindex)
    //println("some try to change the movie datas end")
    val ratingsResourcesPath = this.getClass.getResource("/webroot/ourdatas/user.csv")

    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)


    val ratingSamplesWithLabel = addSampleLabel(ratingSamples)

    val samplesWithMovieFeatures = addMovieFeatures(movieSamples, ratingSamplesWithLabel)

    val samplesWithUserFeatures = addUserFeatures(samplesWithMovieFeatures)
    //save samples as csv format
    splitAndSaveTrainingTestSamples(samplesWithUserFeatures, "/webroot/test0427")
    //save user features and item features to redis for online inference
    extractAndSaveUserFeaturesToRedis(samplesWithUserFeatures)
    //println("before save to redis")
    extractAndSaveMovieFeaturesToRedis(samplesWithUserFeatures)
    spark.close()
  }

}
