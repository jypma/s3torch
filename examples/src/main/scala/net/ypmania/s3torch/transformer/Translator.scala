package net.ypmania.s3torch.transformer

import net.ypmania.s3torch.Batcher
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.DType.Bool
import net.ypmania.s3torch.Default
import net.ypmania.s3torch.Default.cuda
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Dim.|/
import net.ypmania.s3torch.HeapExternal.scoped
import net.ypmania.s3torch.Index
import net.ypmania.s3torch.Shape.Select.First
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.nn.CrossEntropy
import net.ypmania.s3torch.optim.Adam
import net.ypmania.s3torch.tokenizer.*
import net.ypmania.s3torch.token.Token32Type
import org.json4s._
import org.json4s.native.JsonMethods.parse

import java.io.File
import scala.annotation.nowarn
import scala.io.Source

case object Src extends Token32Type
case object Dst extends Token32Type

class Translator[
  SequenceLength <: Dim,
  DModel <: Dim,
  DFF <: Dim,
  NHeads <: Dim,
  Dv <: Device
](sequenceLength: SequenceLength, dModel: DModel, dff: DFF, nHeads: NHeads, srcData: WordData[Src.S], dstData: WordData[Dst.S])(using Default[Dv], DModel |/ NHeads) {
  object src extends WordTokenizer[Src.S](srcData) {
    val sos = reservedToken
    val eos = reservedToken
    val pad = reservedToken
  }
  object dst extends WordTokenizer[Dst.S](dstData) {
    val sos = reservedToken
    val eos = reservedToken
    val pad = reservedToken
  }
  println(s"dst: ${dst.sos} ${dst.eos} ${dst.pad}")
  case object SrcVocabSize extends Dim.Dynamic(src.max.value)
  case object DstVocabSize extends Dim.Dynamic(dst.max.value)
  val model = Transformer(SrcVocabSize, DstVocabSize, sequenceLength, sequenceLength, dModel, dff, nHeads, /*layers*/ 6)

  type Tokens[T <: DType] = Tensor[SequenceLength *: EmptyTuple, T, Dv]

  case class Example(srcText: String, dstText: String, encoderInput: Tokens[Src.DType], decoderInput: Tokens[Dst.DType], label: Tokens[Dst.DType]) {
    def encoderMask = (encoderInput #!= src.pad)
    def decoderMask = (decoderInput #!= dst.pad) && causalMask(sequenceLength) // TODO investigate need for .unsqueeze(1) to add batchSize

    override def toString = s"${srcText} -> ${dstText}"
  }

  object Example {
    def apply(srcText: String, dstText: String): Option[Example] = {
      val srcTok = src.tokenize(srcText)
      val dstTok = dst.tokenize(dstText)

      for {
        encoderInput <- Tensor(src.sos +: srcTok :+ src.eos).padToOption(sequenceLength, src.pad)
        decoderInput <- Tensor(dst.sos +: dstTok).padToOption(sequenceLength, dst.pad)
        label <- Tensor(dstTok :+ dst.eos).padToOption(sequenceLength, dst.pad)
      } yield new Example(srcText, dstText, encoderInput, decoderInput, label)
    }
  }

  def train(batchSize: Int, trainingData: Iterable[Example], validationData: Iterable[Example])(step: => Unit): Unit = {
    validate(validationData)
    var count = 0
    case class BatchSize(size: Long) extends Dim
    val batches = trainingData.grouped(batchSize).toSeq.map(g => Batcher(BatchSize(_), g))
    var finalLoss: Float = 0
    for (batch <- batches) {
      model.train(true)
      scoped {
        val start = System.nanoTime()
        count += 1
        val encoderInput = batch(_.encoderInput)
        val decoderInput = batch(_.decoderInput)
        val label = batch(_.label)
        val encoderMask = batch { x =>
          // We need to add SeqLen and NHeads to match the attention scores (Batch, NHeads, SeqLen, SeqLen).
          val r = x.encoderMask.unsqueezeBefore(First).unsqueezeBefore(First)
          // Somehow, doesn't compile when inlined.
          r
        }
        val decoderMask = batch { x =>
          // We need to add NHeads to match the attention scores (Batch, NHeads, SeqLen, SeqLen).
          //println("decoderMask: " + x.decoderMask.to(Device.CPU).value.map(_.mkString(",")).mkString("\n"))
          val r = x.decoderMask.unsqueezeBefore(First)
          r
        }

        val encoderOutput = scoped { model.encode(encoderInput, encoderMask) }
        val decoderOutput = scoped { model.decode(encoderOutput, encoderMask, decoderMask)(decoderInput) }
        val projOutput = scoped { model.project(decoderOutput) }

        // Let's merge the SequenceLength dimension into BatchSize, so we can do a cross entropy loss of
        // all examples in the batch.
        val expected = label.view.merge[SequenceLength]
        //println("  expected: " + expected.summary)
        val actual = projOutput.view.merge[SequenceLength]
        //println("  actual (first 10): " + actual(Index.All, Index.Slice(0, 10)).summary)
        // TODO see if we can make "ignoreIndex" typesafe
        val loss = CrossEntropy(actual, expected.to(DType.int64), ignoreIndex = Some(dst.pad.value), labelSmoothing = 0.1)

        val end = System.nanoTime()
        finalLoss = loss.to(Device.CPU).value
        //println(s"Batch ${count} of ${batches.size}:  loss is ${loss.to(Device.CPU).value}, took ${(end - start)/1000000}ms")
        if (loss.isNan.sum.to(Device.CPU).value) {
          throw new RuntimeException("Loss became NaN")
        }
        loss.backward()
        step
        if (count % 500 == 0) {
          validate(validationData)
        }
      }
    }
    println("Loss = " + finalLoss)
  }

  def validate(examples: Iterable[Example]): Unit = {
    model.train(false)
    Tensor.noGrad {
      for (x <- examples) {
        scoped {
          // TODO refactor encode and decode to use Batched, so we can remove the batch dimension here entirely.
          val encoderInput = x.encoderInput
            .unsqueezeBefore(First) // add BatchSize of 1
          val encoderMask = x.encoderMask
            .unsqueezeBefore(First).unsqueezeBefore(First).unsqueezeBefore(First) // add NHeads, SeqLen, BatchSize

          // Note: start of greedy_decode
          val source = encoderInput
          val sourceMask = encoderMask
          val encoderOutput = model.encode(source, sourceMask)
          class InputSequenceLength(size: Long) extends Dim.Dynamic(size)
          // TODO see if we can get the tokenizing pattern (start and end tokens) extracted to a type class
          var decoderInput = Tensor(dst.sos :: Nil).shaped[InputSequenceLength]
          def inputLength = decoderInput.sizeOf(InputSequenceLength(_))

          while (inputLength <= sequenceLength && !decoderInput(Index.Last).equal(Tensor(dst.eos))) {
            val nextToken = scoped {
              val decoderMask = causalMask(inputLength)
              val out = model.decode(encoderOutput, sourceMask, decoderMask)(decoderInput.unsqueezeBefore(First)) // Add BatchSize

              // TODO investigate Dim -> Index tuple syntax here, so we can remove the comment
              val in = out(Index.First, Index.Last, Index.All) // Grab the First (only) batch, and only the  Last token in that batch
              val prob = model.project(in)
              val p = prob.to(Device.CPU).value.toSeq.zipWithIndex.sortBy(_._1).reverse.take(10)
              val next = prob.maxBy(DstVocabSize).indices.to(Dst.dType).unsqueeze
              println("  probs: " + p.mkString(",") + " -> " + dst.untokenize(next.to(Device.CPU).value.toSeq))
              next
            }
            // TODO introduce NotGiven[IsIndex[DType]] to disallow += here
            decoderInput ++= nextToken
          }
          // Note: end of greedy_decode

          val value = decoderInput.to(Device.CPU).value.toSeq
          val modelOut = dst.untokenize(value.drop(1) /* remove start of sentence token */ )
          println(s"${x}: ${modelOut}")
        }
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
  val endEpoch = 1000
  val learningRate = 1e-6

  /*
  case object SequenceLength extends Dim.Static[64L]
  case object DModel extends Dim.Static[32L]
  case object DFF extends Dim.Static[32L]
  case object NHeads extends Dim.Static[8L]
  val layers = 6
  val batchSize = 48
  val endEpoch = 2000
  val learningRate = 1e-4
   */
  @main def run(): Unit = {
    val srcLang = "en"
    val dstLang = "nl"
    val baseFile = s"model_${srcLang}_${dstLang}_s${SequenceLength.size}_m${DModel.size}_d${DFF.size}_h${NHeads.size}"
    def modelFile(epoch: Int) = s"${baseFile}_e${epoch}.pt"
    def optFile(epoch: Int) = s"${baseFile}_e${epoch}_optimizer.pt"

    val en_nl = translations(srcLang, dstLang)
    // TODO save and auto-load tokenizers
    val translator = new Translator(SequenceLength, DModel, DFF, NHeads,
      WordTokenizer.train[Src.S](en_nl.map(_._1)),
      WordTokenizer.train[Dst.S](en_nl.map(_._2))
    )
    val allExamples = en_nl.flatMap(translator.Example(_, _))
    // Note: python original using LR of 1e-4, but that's all over the place. Let's use 1e-6 and be patient.
    val optimizer = Adam(translator.model.parameters, learningRate = learningRate, eps = 1e-9)

    val startEpoch = 0.to(endEpoch).reverse.find(e => new File(modelFile(e)).exists()).map(e => e + 1).getOrElse(0)
    if (startEpoch > 0) {
      val lastEpoch = startEpoch - 1
      println(s"Loading epoch ${startEpoch}")
      translator.model.load(modelFile(lastEpoch))
      optimizer.load(optFile(lastEpoch))
    }
    for (epoch <- startEpoch.until(endEpoch)) {
      val indexes = Tensor.randperm(Dim(allExamples.size))(using Default.int32, Default.cpu).value.toSeq
      val splitIdx = (indexes.size * 0.9).toInt
      val trainingData = indexes.take(splitIdx).map(allExamples(_))
      val validationData = indexes.drop(splitIdx).take(1).map(allExamples(_))
      println(s"Epoch ${epoch}")
      translator.train(batchSize, trainingData, validationData) {
        optimizer.step()
        optimizer.zeroGrad()
        //Thread.sleep(1000)
      }
      translator.model.save(modelFile(epoch))
      optimizer.save(optFile(epoch))
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
