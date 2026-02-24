package net.ypmania.s3torch.transformer

import org.json4s._
import org.json4s.native.JsonMethods.parse
import scala.io.Source

class Translator() {

}

object Translator {
  @main def run(): Unit = {
    val en_nl = trainingData("en", "nl")
    val enDict = WordTokenizer.train(en_nl.map(_._1), 2) // minCount 2 reduces dictionary by 50%
    val nlDict = WordTokenizer.train(en_nl.map(_._2), 2)
    println(enDict.known.size)
    println(nlDict.known.size)
    println(enDict.tokenize(en_nl(42)._1))
  }

  def trainingData(from: String, to: String): Seq[(String, String)] = {
    implicit val formats: Formats = DefaultFormats

    Source.fromFile(s"src/test/resources/${from}_${to}.ndjson").getLines.map { line =>
      val json = parse(line) \ "translation"
      ((json \ from).extract[String], (json \ to).extract[String])
    }.toVector
  }
}
