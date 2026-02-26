package net.ypmania.s3torch.transformer

import org.json4s._
import org.json4s.native.JsonMethods.parse
import scala.io.Source
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.DType.Int32
import net.ypmania.s3torch.Default
import net.ypmania.s3torch.PaddingMode.Append
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.DType.Bool

class Translator[
  SequenceLength <: Dim,
  Dv <: Device
](sequenceLength: SequenceLength, srcTok: Tokenizer, dstTok: Tokenizer)(using Default[Dv]) {
  val srcStart = srcTok.max + 1
  val srcEnd = srcTok.max + 2
  val srcPad = srcTok.max + 3

  val dstStart = dstTok.max + 1
  val dstEnd = dstTok.max + 2
  val dstPad = dstTok.max + 3

  type Tokens = Tensor[SequenceLength *: EmptyTuple, Int32.type, Dv]

  case class Example(encoderInput: Tokens, decoderInput: Tokens, label: Tokens) {
    def encoderMask = encoderInput #!= srcPad // TODO investigate need for twice .unsqueeze(1) to add sequenceLength and batchSize
    def decoderMask = (decoderInput #!= dstPad) && causalMask(sequenceLength) // TODO investigate need for .unsqueeze(1) to add batchSize
  }
  object Example {
    def apply(src: Seq[Int], dst: Seq[Int]): Example = {
      val encoderInput = Tensor(srcStart +: src :+ srcEnd).padTo(sequenceLength)(srcPad, Append)
      val decoderInput = Tensor(dstStart +: dst).padTo(sequenceLength)(dstPad, Append)
      val label = Tensor(src :+ srcEnd).padTo(sequenceLength)(srcPad, Append)

      ???
    }
  }

  def causalMask[D <: Dim](dim: D): Tensor[(D, D), Bool.type, Dv] = {
    Tensor.ones(using Default(DType.Int32))(dim, dim).triu(1) #== 0
  }

}

object Translator {
  case object SequenceLength extends Dim.Static[128L]

  @main def run(): Unit = {
    val en_nl = trainingData("en", "nl")
    val enDict = WordTokenizer.train(en_nl.map(_._1), 2) // minCount 2 reduces dictionary by 50%
    // TODO types for input and output tokens, so they don't mix
    val enStart = enDict.max + 1
    val enEnd = enDict.max + 2
    val enPad = enDict.max + 3

    val nlDict = WordTokenizer.train(en_nl.map(_._2), 2)
    val nlStart = nlDict.max + 1
    val nlEnd = nlDict.max + 2
    val nlPad = nlDict.max + 3

    val data = en_nl
      .map((en, nl) =>
        (enDict.tokenize(en), nlDict.tokenize(nl))
      ).filter((en, nl) =>
        en.size <= SequenceLength.size && nl.size <= SequenceLength.size
      )
      .take((en_nl.size * 0.9).toInt).map { (en, nl) =>
        val encoderInput = (enStart +: en :+ enEnd).padTo(SequenceLength.size.toInt, enPad)
        val decoderInput = (nlStart +: nl).padTo(SequenceLength.size.toInt, nlPad)
        val label = (nl :+ nlEnd).padTo(SequenceLength.size.toInt, nlPad)
      }
  }

  def trainingData(from: String, to: String): Seq[(String, String)] = {
    implicit val formats: Formats = DefaultFormats

    Source.fromFile(s"src/test/resources/${from}_${to}.ndjson").getLines.map { line =>
      val json = parse(line) \ "translation"
      ((json \ from).extract[String], (json \ to).extract[String])
    }.toVector
  }
}
