package coursera.recommendation.course2

import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ArrayBuffer

object UserUserCollaborationFiltering {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("course2.week2")
      .master("local[2]")
      .config("spark.sql.crossJoin.enabled", true)
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()

    val path = "src/main/resources/UUCF_Assignment_Spreadsheet.csv"
    val fields = extractFields(spark, path)
    val data = loadData(spark, path, fields)

    spark
      .sql(
      """
        |SELECT movie, _3867, _89
        |FROM movies_users
        |WHERE _3867 IS NOT NULL OR _89 IS NOT NULL
        |ORDER BY _3867, _89
      """.stripMargin)
      .show(100, false)

    printResult(spark, data, fields, "_3867")
    printResult(spark, data, fields, "_89")
  }

  def printResult(spark: SparkSession, data: DataFrame, fields: ArrayBuffer[StructField], targetUser: String): Unit = {
    val neighbours = findFiveBestCorrs(spark, targetUser, fields, data)
    val recommendations = recommend(spark, data, neighbours)
    val normaRecommendations = recommendNormalized(spark, data, targetUser, neighbours, true)
    println(s"-----------$targetUser-----------")
    println(recommendations.mkString("\n"))
    println(s"$targetUser RMSE: ", computeRMSE(spark, targetUser, recommendations))

    println(s"-----------$targetUser normalization rec-----------")
    println(normaRecommendations.mkString("\n"))
    println(s"$targetUser normalized RMSE: ", computeRMSE(spark, targetUser, normaRecommendations))
  }

  def extractFields(spark: SparkSession, path: String): ArrayBuffer[StructField] = {
    val firstRow = spark
      .read
      .csv(path)
      .first()

    val fields = new ArrayBuffer[StructField]()
    var i = 0
    while (i < firstRow.size) {
      if (i == 0) {
        fields += StructField(firstRow.getAs[String](i), StringType, false)
      } else {
        val column = "_" + firstRow.getAs[String](i)
        fields += StructField(column, DoubleType, true)
      }

      i += 1
    }
    fields
  }

  def loadData(spark: SparkSession, path: String, fields: ArrayBuffer[StructField]): DataFrame = {
    val schema = StructType(fields)
    val data = spark
      .read
      .schema(schema)
      .option("header", true)
      .csv(path)
      .cache()

    data.createOrReplaceTempView("movies_users")
    data
  }

  def findFiveBestCorrs(spark: SparkSession,
                        targetColumn: String,
                        fields: ArrayBuffer[StructField],
                        data: DataFrame): Map[String, (Double, Double)] = {
    import spark.implicits._

    val fiveBestCorrs = new ArrayBuffer[(String, (Double, Double))]()
    for (field <- fields) {
      val column = field.name
      if (targetColumn != column && column != "movie") {
        val stats = data
          .selectExpr(s"""corr($targetColumn, $column) AS corr""", s"AVG($column) AS average")
          .map(row => (row.getAs[Double]("corr"), row.getAs[Double]("average")))
          .first()

        val targetDS = data
          .na.fill(-1000, Seq(targetColumn))
          .select(targetColumn)
          .map { row =>
            row.getDouble(0)
          }
          .rdd
        val neighbourDS = data
          .na.fill(-1000, Seq(column))
          .select(column)
          .map { row =>
            row.getDouble(0)
          }
          .rdd

        val spearman = Statistics.corr(targetDS, neighbourDS, "spearman")
        fiveBestCorrs += ((column, (spearman, stats._2)))
      }
    }
    fiveBestCorrs
      .sortWith((a, b) => a._2._1.compare(b._2._1) > 0)
      .take(5)
      .toMap
  }

  def recommend(spark: SparkSession,
                data: DataFrame,
                neighbours: Map[String, (Double, Double)],
                norm: Boolean = false): Array[(String, Double)] = {
    val bneighbours = spark.sparkContext.broadcast(neighbours)
    data
      .rdd
      .map { row =>
        val movie = row.getAs[String]("movie")
        var numerator = 0.0
        var denominator = 0.0
        for ((neighbour, (weight, avgRating)) <- bneighbours.value) {
          val rating = Option(row.getAs[Double](neighbour))
          if (rating.isDefined) {
            numerator += rating.get * weight
            denominator += weight
          }
        }

        (movie, numerator / denominator)
      }
      .sortBy(x => x._2, false)
      .take(12)
  }

  def recommendNormalized(spark: SparkSession,
                data: DataFrame,
                targetUser: String,
                neighbours: Map[String, (Double, Double)],
                norm: Boolean = false): Array[(String, Double)] = {
    import spark.implicits._

    val bneighbours = spark.sparkContext.broadcast(neighbours)
    val userAvgRating = data
      .selectExpr(s"AVG($targetUser) AS average")
      .map(row => row.getAs[Double]("average"))
      .first()
    val buserAvgRating = spark.sparkContext.broadcast(userAvgRating)

    data
      .rdd
      .map { row =>
        val movie = row.getAs[String]("movie")
        var numerator = 0.0
        var denominator = 0.0
        for ((neighbour, (weight, avgRating)) <- bneighbours.value) {
          val rating = Option(row.getAs[Double](neighbour))
          if (rating.isDefined) {
            numerator += (rating.get - avgRating) * weight
            denominator += weight
          }
        }

        (movie, buserAvgRating.value + numerator / denominator)
      }
      .sortBy(x => x._2, false)
      .collect()
  }

  def computeRMSE(spark: SparkSession, targetUser: String, pred: Array[(String, Double)]): Double = {
    val trueMovieRatings = spark.sql(
      s"""
        |SELECT movie, $targetUser
        |FROM movies_users
        |WHERE $targetUser IS NOT NULL
      """.stripMargin)
      .collect()
      .map(row => (row.getAs[String]("movie"), row.getAs[Double](targetUser)))
      .toMap

    var sum = 0.0
    var n = 0
    for ((movie, predRating) <- pred) {
      val trueRating = trueMovieRatings.get(movie)
      if (trueRating.isDefined) {
        sum += Math.pow(trueRating.get - predRating, 2)
        n += 1
      }
    }

    Math.sqrt(sum / n)
  }
}
