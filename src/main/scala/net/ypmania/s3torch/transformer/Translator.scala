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
import net.ypmania.s3torch.internal.FromScala.ToScalar

case object Src extends TokenType
case object Dst extends TokenType

class Translator[
  SequenceLength <: Dim,
  Dv <: Device
](sequenceLength: SequenceLength, srcData: WordData[Src.T], dstData: WordData[Dst.T])(using Default[Dv]) {
  object srcDict extends WordTokenizer[Src.T](srcData) {
    val sos = reservedToken
    val eos = reservedToken
    val pad = reservedToken
  }
  object dstDict extends WordTokenizer[Dst.T](dstData) {
    val sos = reservedToken
    val eos = reservedToken
    val pad = reservedToken
  }

  type Tokens[T <: DType] = Tensor[SequenceLength *: EmptyTuple, T, Dv]

  case class Example(encoderInput: Tokens[Src.DType], decoderInput: Tokens[Dst.DType], label: Tokens[Dst.DType]) {
    def encoderMask = encoderInput #!= srcDict.pad // TODO investigate need for twice .unsqueeze(1) to add sequenceLength and batchSize
    def decoderMask = (decoderInput #!= dstDict.pad) && causalMask(sequenceLength) // TODO investigate need for .unsqueeze(1) to add batchSize
  }
  /*
  object Example {
    def apply(src: Seq[Int], dst: Seq[Int]) = new Example (
      encoderInput = Tensor(srcStart +: src :+ srcEnd, SrcToken).padTo(sequenceLength)(srcPad, Append),
      decoderInput = Tensor(dstStart +: dst, DstToken).padTo(sequenceLength)(dstPad, Append),
      label = Tensor(src :+ srcEnd, DstToken).padTo(sequenceLength)(srcPad, Append)
    )
  }
   */
  def causalMask[D <: Dim](dim: D): Tensor[(D, D), Bool, Dv] = {
    Tensor.ones(using Default(DType.int32))(dim, dim).triu(1) #== 0
  }

}

object Translator {
  /*
  case object SequenceLength extends Dim.Static[128L]

  @main def run(): Unit = {
    val en_nl = trainingData("en", "nl")
    object enDict extends WordTokenizer[Int](WordTokenizer.train(en_nl.map(_._1), 2)) {
      val sos = reservedToken
      val eos = reservedToken
      val pad = reservedToken
    }

    val nlDict = WordTokenizer(WordTokenizer.train(en_nl.map(_._2), 2))
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
        val encoderInput = (enDict.sos +: en :+ enDict.eos).padTo(SequenceLength.size.toInt, enDict.pad)
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
   */
}
