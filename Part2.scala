package coursera.ml

import breeze.linalg.Matrix
import org.apache.spark.SparkContext
import org.apache.spark.mllib.stat.Statistics

import scala.collection.mutable

object Part2 {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local[*]", "myapp")
    sc.setLogLevel("ERROR")
    val data = sc.textFile("src/main/resources/HW1-data.csv")

    var i = 0
    val header = data.first()
    val map2 = scala.collection.mutable.Map[Int, Int]()
    println(header)
    for (s <- header.split(",")) {
      if (s.contains(":")) {
        var expression = if (s.charAt(0).equals('"')) s.substring(1) else s
        val idName = expression.split(": ")
        map2.put(i, idName(0).toInt)
        i += 1
      }
    }
    val movies = data.filter(line => !line.startsWith("User"))
      .map(line => line.split(","))
      .flatMap(data => {
        val tuples = new mutable.MutableList[(Int, Int, Int, Int)]()
        val userId = data(0).toInt
        val gender = data(1).toInt
        val ratings = data.slice(2, data.length)
        for (i <- ratings.indices) {
          val rating = if (ratings(i).contentEquals("")) 0 else ratings(i).toInt
          val movieId = map2(i)
          tuples += ((userId, gender, movieId, rating))
        }
        tuples
      })
      .cache()
    // userId, gender, movieId, rating

    val aggregatedMovies = movies.map(x => (x._3, x._4))
      .aggregateByKey((0, 0))(
        (acc, r) => (acc._1 + r, acc._2 + 1),
        (r1, r2) => (r1._1 + r2._1, r1._2 + r2._2))
      .cache()
    println("1---------------Mean Rating")
    aggregatedMovies.mapValues(x => x._1 * 1.0 / x._2)
      .sortBy(x => x._2, ascending = false)
      .take(3)
      .foreach(e => println(e))
    println("2---------------Rating Count (popularity)")
    aggregatedMovies
      .mapValues(x => x._2)
      .sortBy(x => x._2, ascending = false)
      .take(3)
      .foreach(println)
    println("3---------------% of ratings 4+ (liking)")
    movies
      .map(x => (x._3, x._4))
      .mapValues(r => if (r >= 4) 1 else 0)
      .aggregateByKey((0, 0))(
        (acc, r) => (acc._1 + r, acc._2 + 1),
        (r1, r2) => (r1._1 + r2._1, r1._2 + r2._2))
      .mapValues(t => t._1 * 1.0 / t._2)
      .sortBy(x => x._2, ascending = false)
      .take(3)
      .foreach(println)
    println("4---------------top association movies for Toy Story")
    val toyStoryMovieId = 1
    val toyStoryRaters = movies
      .filter(x => x._3 == toyStoryMovieId)
      .map(x => x._1)
      .collect()
    val toyStoryCount = toyStoryRaters.length
    val broadcastToyStoryRaters = sc.broadcast(toyStoryRaters)

    movies
      .filter(x => broadcastToyStoryRaters.value.contains(x._1))
      .map(x => (x._3, 1))
      .reduceByKey(_ + _)
      .mapValues(count => count * 1.0 / toyStoryCount)
      .sortBy(x => x._2, ascending = false)
      .take(5)
      .foreach(println)
    println("5---------------")
    val toyStoryRDD = movies
      .filter(x => x._3 == toyStoryMovieId)
      .map(x => x._4.toDouble)

    val otherMovies = movies
      .filter(x => x._3 != toyStoryMovieId)
      .map(x => (x._3, x._4.toDouble))
      .groupByKey()
      .collectAsMap()

    otherMovies.foreach(x => {
//      Matrix.
      val otherMovieRatings = sc.parallelize(x._2.toSeq)
      val corr = Statistics.corr(toyStoryRDD, otherMovieRatings)
      println(s"toy story movie: other movie-${x._1} = $corr")
    })


//    Statistics.corr(, , "pearson")
    println("6---------------mean rating difference by gender")
    val meanFemalesMales = movies
      .map(x => ((x._2, x._3), x._4))
      .aggregateByKey((0, 0))((acc, r) => (acc._1 + r, acc._2 + 1), (acc1, acc2) => (acc1._1 + acc2._1, acc1._2 + acc2._2))
      .mapValues(x => x._1 * 1.0 / x._2)
      .map(x => (x._1._2, (x._1._1, x._2)))
      .aggregateByKey((0, 0.0))((acc, v) => {
        val value = if (v._1 == 1) acc._2 + v._2 else acc._2 - v._2
        val result = if (value > 0) (1, value) else (0, value)
        result
      }, (acc1, acc2) => {
        val result = acc1._2 + acc2._2
        if (result > 0) (1, result) else (0, result)
      })
      .sortBy(x => x._2._2)
      .collect()
    println(meanFemalesMales(0), meanFemalesMales(meanFemalesMales.length - 1))

    val overallAverageRating = movies
      .map(x => (x._2, x._4))
      .aggregateByKey((0, 0))((acc, r) => (acc._1 + r, acc._2 + 1), (acc1, acc2) => (acc1._1 + acc2._1, acc1._2 + acc2._2))
      .mapValues(t => t._1 * 1.0 / t._2)
      .reduce((x, y) => {
        val result = if (x._1 == 1) x._2 - y._2 else y._2 - x._2
        if (result > 0) (1, result) else (0, result)
      })
    println(overallAverageRating)
    println("7---------------up votes mean rating difference by gender")
    val meanUpFemalesMales = movies
      .map(x => ((x._2, x._3), x._4))
      .mapValues(r => if (r >= 4) 1 else 0)
      .aggregateByKey((0, 0))((acc, r) => (acc._1 + r, acc._2 + 1), (acc1, acc2) => (acc1._1 + acc2._1, acc1._2 + acc2._2))
      .mapValues(x => x._1 * 1.0 / x._2)
      .map(x => (x._1._2, (x._1._1, x._2)))
      .aggregateByKey((0, 0.0))((acc, v) => {
        val value = if (v._1 == 1) acc._2 + v._2 else acc._2 - v._2
        val result = if (value > 0) (1, value) else (0, value)
        result
      }, (acc1, acc2) => {
        val result = acc1._2 + acc2._2
        if (result > 0) (1, result) else (0, result)
      })
      .sortBy(x => x._2._2)
      .collect()
    println(meanUpFemalesMales(0), meanUpFemalesMales(meanUpFemalesMales.length - 1))
  }
}
