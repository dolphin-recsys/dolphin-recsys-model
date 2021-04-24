package com.recsys.dolphin.offline

import org.apache.spark.SparkConf
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

/**
 * @author kris
 * 2021/04/24
 */
object CollaborativeFiltering {
    def main(args: Array[String]): Unit = {
        val conf: SparkConf = new SparkConf()
            .setMaster("local")
            .setAppName("collaborativeFiltering")
            .set("spark.submit.deployMode", "client")
        // create sparkSession
        val spark: SparkSession = SparkSession.builder.config(conf).getOrCreate()

        // use als to build recommendation model
        val als: ALS = new ALS()
            .setMaxIter(5)
            .setRegParam(0.01)
            .setUserCol("userIdInt")
            .setItemCol("movieIdInt")
            .setRatingCol("ratingFloat")

        spark.stop()
    }
}
