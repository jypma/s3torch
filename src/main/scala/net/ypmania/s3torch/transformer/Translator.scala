package net.ypmania.s3torch.transformer

import org.json4s._
import org.json4s.native.JsonMethods.parse
import scala.io.Source
import net.ypmania.s3torch.Batcher
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.DType.Int32
import net.ypmania.s3torch.Default
import net.ypmania.s3torch.PaddingMode.Append
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Shape.Select.First
import net.ypmania.s3torch.DType.Bool
import net.ypmania.s3torch.internal.FromScala.ToScalar
import scala.annotation.nowarn
import net.ypmania.s3torch.optim.Adam
import net.ypmania.s3torch.nn.CrossEntropy

case object Src extends TokenType
case object Dst extends TokenType

class Translator[
  SequenceLength <: Dim,
  Dv <: Device
](sequenceLength: SequenceLength, srcData: WordData[Src.T], dstData: WordData[Dst.T])(using Default[Dv]) {
  object src extends WordTokenizer[Src.T](srcData) {
    val sos = reservedToken
    val eos = reservedToken
    val pad = reservedToken
  }
  object dst extends WordTokenizer[Dst.T](dstData) {
    val sos = reservedToken
    val eos = reservedToken
    val pad = reservedToken
  }

  type Tokens[T <: DType] = Tensor[SequenceLength *: EmptyTuple, T, Dv]

  case class Example(encoderInput: Tokens[Src.DType], decoderInput: Tokens[Dst.DType], label: Tokens[Dst.DType]) {
    def encoderMask = (encoderInput #!= src.pad)
    def decoderMask = (decoderInput #!= dst.pad) && causalMask(sequenceLength) // TODO investigate need for .unsqueeze(1) to add batchSize
  }

  object Example {
    def apply(srcText: String, dstText: String): Option[Example] = {
      val srcTok = src.tokenize(srcText)
      val dstTok = dst.tokenize(dstText)

      for {
        encoderInput <- Src.toTensor(src.sos +: srcTok :+ src.eos, sequenceLength, src.pad)
        decoderInput <- Dst.toTensor(dst.sos +: dstTok, sequenceLength, dst.pad)
        label <- Dst.toTensor(dstTok :+ dst.eos, sequenceLength, dst.pad)
      } yield new Example(encoderInput, decoderInput, label)
    }
  }

  def causalMask[D <: Dim](dim: D): Tensor[(D, D), Bool, Dv] = {
    Tensor.ones(using Default(DType.int32))(dim, dim).triu(1) #== 0
  }
}

object Translator {
  case object SequenceLength extends Dim.Static[128L]

  @main def run(): Unit = {
    val en_nl = translations("en", "nl")
    // TODO save and auto-load tokenizers
    // TODO investigate tokenizer lack of performance
    val translator = new Translator(SequenceLength,
      WordTokenizer.train[Src.T](en_nl.map(_._1)),
      WordTokenizer.train[Dst.T](en_nl.map(_._2))
    )
    val allExamples = en_nl.flatMap(translator.Example(_, _))
    case class BatchSize(size: Long) extends Dim
    case object SrcVocabSize extends Dim.Dynamic(translator.src.size)
    case object DstVocabSize extends Dim.Dynamic(translator.dst.size)
    val model = Transformer(SrcVocabSize, DstVocabSize, SequenceLength, SequenceLength)
    val optimizer = Adam(model.parameters, learningRate = 1e-4, eps = 1e-9)

    val trainingData = allExamples.take((allExamples.size * 0.9).toInt)
    // TODO save after each epoch, resume
    for (epoch <- 0.until(20)) {
      model.train(true)

      for (batch <- trainingData.grouped(64).map(g => Batcher(BatchSize(_), g))) {
        val encoderInput = batch(_.encoderInput).to(DType.int32) // TODO type safety on the DType in Transformer.encode
        val decoderInput = batch(_.decoderInput).to(DType.int32) // TODO type safety on the DType in Transformer.decode
        val label = batch(_.label)
        val encoderMask = batch { x =>
          // We need to add dimensions to match the attention scores (Batch, NHeads, SeqLen, SeqLen).
          val r = x.encoderMask.unsqueezeBefore(First).unsqueezeBefore(First)
          // Somehow, doesn't compile when inlined.
          r
        }.to(DType.float32) // TODO investigate Bool for mask type
        val decoderMask = batch { x =>
          // We need to add dimensions to match the attention scores (Batch, NHeads, SeqLen, SeqLen).
          val r = x.decoderMask.unsqueezeBefore(First)
          r
        }.to(DType.float32)  // TODO investigate Bool for mask type

        val encoderOutput = model.encode(encoderInput, encoderMask)
        val decoderOutput = model.decode(encoderOutput, encoderMask, decoderMask)(decoderInput)
        val projOutput = model.project(decoderOutput)

        //val loss = CrossEntropy(ignoreIndex = Some(translator.src.pad.toInt), labelSmoothing = 0.1)
      }
    }
  }

  @nowarn // https://github.com/json4s/json4s/issues/982
  def translations(from: String, to: String): Seq[(String, String)] = {
    implicit val formats: Formats = DefaultFormats

    Source.fromFile(s"src/test/resources/${from}_${to}.ndjson").getLines.map { line =>
      val json = parse(line) \ "translation"
      ((json \ from).extract[String], (json \ to).extract[String])
    }.toVector
   }
}
