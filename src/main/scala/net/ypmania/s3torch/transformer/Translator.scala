package net.ypmania.s3torch.transformer

import net.ypmania.s3torch.Batcher
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.DType.Bool
import net.ypmania.s3torch.Default
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Dim.|/
import net.ypmania.s3torch.Shape.Select.First
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.HeapExternal.scoped
import net.ypmania.s3torch.optim.Adam
import org.json4s._
import org.json4s.native.JsonMethods.parse

import scala.annotation.nowarn
import scala.io.Source
import net.ypmania.s3torch.nn.CrossEntropy

import net.ypmania.s3torch.Default.cuda
import java.io.File
import net.ypmania.s3torch.Default

case object Src extends IntTokenType
case object Dst extends IntTokenType

class Translator[
  SequenceLength <: Dim,
  DModel <: Dim,
  DFF <: Dim,
  NHeads <: Dim,
  Dv <: Device
](sequenceLength: SequenceLength, dModel: DModel, dff: DFF, nHeads: NHeads, srcData: WordData[Src.T], dstData: WordData[Dst.T])(using Default[Dv], DModel |/ NHeads) {
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
  case object SrcVocabSize extends Dim.Dynamic(src.max.toInt)
  case object DstVocabSize extends Dim.Dynamic(dst.max.toInt)
  val model = Transformer(SrcVocabSize, DstVocabSize, sequenceLength, sequenceLength, dModel, dff, nHeads, /*layers*/ 6)

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

  def train(batchSize: Int, trainingData: Iterable[Example]): Unit = {
    model.train(true)

    var count = 0
    case class BatchSize(size: Long) extends Dim
    val batches = trainingData.grouped(batchSize).toSeq.map(g => Batcher(BatchSize(_), g))
    for (batch <- batches) {
      scoped {
        val start = System.nanoTime()
        count += 1
        val encoderInput = batch(_.encoderInput).to(DType.int32) // TODO type safety on the DType in Transformer.encode
        val decoderInput = batch(_.decoderInput).to(DType.int32) // TODO type safety on the DType in Transformer.decode
        val label = batch(_.label)
        val encoderMask = batch { x =>
          // We need to add SeqLen and NHeads to match the attention scores (Batch, NHeads, SeqLen, SeqLen).
          val r = x.encoderMask.unsqueezeBefore(First).unsqueezeBefore(First)
          // Somehow, doesn't compile when inlined.
          r
        }.to(DType.float32) // TODO investigate Bool for mask type
        val decoderMask = batch { x =>
          // We need to add NHeads to match the attention scores (Batch, NHeads, SeqLen, SeqLen).
          val r = x.decoderMask.unsqueezeBefore(First)
          r
        }.to(DType.float32)  // TODO investigate Bool for mask type

        val encoderOutput = scoped { model.encode(encoderInput, encoderMask) }
        val decoderOutput = scoped { model.decode(encoderOutput, encoderMask, decoderMask)(decoderInput) }
        val projOutput = scoped { model.project(decoderOutput) }

        // Let's merge the SequenceLength dimension into BatchSize, so we can do a cross entropy loss of
        // all examples in the batch.
        val expected = label.view.merge[SequenceLength]
        /*
        val actual = projOutput.view.merge[SequenceLength]
        val loss = CrossEntropy(actual, expected.to(DType.int64), ignoreIndex = Some(src.pad.toInt), labelSmoothing = 0.1)

        val end = System.nanoTime()
        println(s"Batch ${count} of ${batches.size}:  loss is ${loss.to(Device.CPU).value}, took ${(end - start)/1000000}ms")
        if (loss.isNan.sumAll.to(Device.CPU).value) {
          throw new RuntimeException("Loss became NaN")
        }
         loss.backward()
         */
        //optimizer.step()
        //optimizer.zeroGrad()
      }
    }
  }

  private def causalMask[D <: Dim](dim: D): Tensor[(D, D), Bool, Dv] = {
    Tensor.ones(using Default(DType.int32))(dim, dim).triu(1) #== 0
  }
}

object Translator {
  // OK: 128 / 512 / 1024 / 8 / 6 / 16 (17ms per batch)
  org.bytedeco.pytorch.global.torch.manual_seed(42)

  case object SequenceLength extends Dim.Static[128L]
  case object DModel extends Dim.Static[512L]
  case object DFF extends Dim.Static[1024L]
  case object NHeads extends Dim.Static[8L]
  val layers = 6
  val batchSize = 16
  val endEpoch = 20

  @main def run(): Unit = {
    /*
    val srcLang = "en"
    val dstLang = "nl"
    val baseFile = s"model_${srcLang}_${dstLang}_s${SequenceLength.size}_m${DModel.size}_d${DFF.size}_h${NHeads.size}"
    def modelFile(epoch: Int) = s"${baseFile}_e${epoch}.pt"
    def optFile(epoch: Int) = s"${baseFile}_e${epoch}_optimizer.pt"

    val en_nl = translations(srcLang, dstLang)
    // TODO save and auto-load tokenizers
    val translator = new Translator(SequenceLength,
      WordTokenizer.train[Src.T](en_nl.map(_._1)),
      WordTokenizer.train[Dst.T](en_nl.map(_._2))
    )
    val allExamples = en_nl.flatMap(translator.Example(_, _))
    case class BatchSize(size: Long) extends Dim
    case object SrcVocabSize extends Dim.Dynamic(translator.src.max.toInt)
    case object DstVocabSize extends Dim.Dynamic(translator.dst.max.toInt)
    val model = Transformer(SrcVocabSize, DstVocabSize, SequenceLength, SequenceLength, DModel, DFF, NHeads, layers)
    val optimizer = Adam(model.parameters, learningRate = 1e-6, eps = 1e-9)

    def validate(examples: Seq[translator.Example]): Unit = {
      model.train(false)
      Tensor.noGrad {
        for (x <- examples) {
          val encoderInput = x.encoderInput.to(DType.int32)  // TODO type safety on the DType in Transformer.encode
            .unsqueezeBefore(First) // add BatchSize of 1
          val encoderMark = x.encoderMask
            .unsqueezeBefore(First).unsqueezeBefore(First).unsqueezeBefore(First) // add NHeads, SeqLen, BatchSize

          // Note: start of greedy_decode

        }
      }
    }

    val startEpoch = 0.to(endEpoch).reverse.find(e => new File(modelFile(e)).exists()).map(e => e + 1).getOrElse(0)
    if (startEpoch > 0) {
      val lastEpoch = startEpoch - 1
      println(s"Loading epoch ${startEpoch}")
      model.load(modelFile(lastEpoch))
      optimizer.load(optFile(lastEpoch))
    }
    for (epoch <- startEpoch.until(endEpoch)) {
      val indexes = Tensor.randperm(Dim(allExamples.size))(using Default.int32, Default.cpu).value.toSeq
      val trainingData = indexes.take((indexes.size * 0.9).toInt).map(idx => allExamples(idx))
      println(s"Epoch ${epoch}")
      //println(model.summary)
      model.train(true)

      var count = 0
      val batches = trainingData.grouped(batchSize).toSeq.map(g => Batcher(BatchSize(_), g))
      for (batch <- batches) {a
        scoped {
          val start = System.nanoTime()
          count += 1
          val encoderInput = batch(_.encoderInput).to(DType.int32) // TODO type safety on the DType in Transformer.encode
          val decoderInput = batch(_.decoderInput).to(DType.int32) // TODO type safety on the DType in Transformer.decode
          val label = batch(_.label)
          val encoderMask = batch { x =>
            // We need to add SeqLen and NHeads to match the attention scores (Batch, NHeads, SeqLen, SeqLen).
            val r = x.encoderMask.unsqueezeBefore(First).unsqueezeBefore(First)
            // Somehow, doesn't compile when inlined.
            r
          }.to(DType.float32) // TODO investigate Bool for mask type
          val decoderMask = batch { x =>
            // We need to add NHeads to match the attention scores (Batch, NHeads, SeqLen, SeqLen).
            val r = x.decoderMask.unsqueezeBefore(First)
            r
          }.to(DType.float32)  // TODO investigate Bool for mask type

          val encoderOutput = scoped { model.encode(encoderInput, encoderMask) }
          val decoderOutput = scoped { model.decode(encoderOutput, encoderMask, decoderMask)(decoderInput) }
          val projOutput = scoped { model.project(decoderOutput) }

          // Let's merge the SequenceLength dimension into BatchSize, so we can do a cross entropy loss of
          // all examples in the batch.
          val expected = label.view.merge[SequenceLength.type]
          val actual = projOutput.view.merge[SequenceLength.type]
          val loss = CrossEntropy(actual, expected.to(DType.int64), ignoreIndex = Some(translator.src.pad.toInt), labelSmoothing = 0.1)

          val end = System.nanoTime()
          println(s"Batch ${count} of ${batches.size}:  loss is ${loss.to(Device.CPU).value}, took ${(end - start)/1000000}ms")
          if (loss.isNan.sumAll.to(Device.CPU).value) {
            throw new RuntimeException("Loss became NaN")
          }
          loss.backward()
          optimizer.step()
          optimizer.zeroGrad()
        }
      }
      model.save(modelFile(epoch))
      optimizer.save(optFile(epoch))
     }
     */
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
