package net.ypmania.s3torch.shakespeare

import net.ypmania.s3torch.tokenizer.*
import net.ypmania.s3torch.token.Token64Type
import scala.io.Source
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Select.dim
import net.ypmania.s3torch.Dim.|<=
import net.ypmania.s3torch.Select.Idx
import net.ypmania.s3torch.Index.Take
import net.ypmania.s3torch.optim.AdamW
import net.ypmania.s3torch.Device.CPU
import net.ypmania.s3torch.Device.CUDA
import net.ypmania.s3torch.HeapExternal.scoped
import net.ypmania.s3torch.Default
import net.ypmania.s3torch.Dim.Ref
import net.ypmania.s3torch.Dim.KnownLessThan
import net.ypmania.s3torch.Control.whileDefined

import net.ypmania.s3torch.Default.cuda
type Dv = CUDA.type
//type Dv = CPU.type

object ShakespeareGen {
  case object Tok extends Token64Type
  case object BlockSize extends Dim.Static[8L]
  class SequenceSize(size: Long) extends Dim.Dynamic(size)

  type TokT[S <: Tuple] = Tensor[S, Tok.DType, Dv]

  // for Bigram, ~2.4 loss
  // for SelfAttention, 2.7 loss after 40000
  // for SelfAttention1, 2.7 loss after 40000, but drops faster
  // for SelfAttention2, 2.61 loss after 40000. 2.4 after 100000. 2.28 after 250000. Somehow 2.2 in video after only 4000...
  case object BatchSize extends Dim.Static[64L]
  val trainingRounds = 250000
  val printLossEvery = 1000
  val learningRate = 1e-5 // SelfAttention1
  // val learningRate = 1e-5 // SelfAttention
  // val learningRate = 1e-4 // Bigram

  case class Batch(
    x: TokT[(BatchSize.type, BlockSize.type)],
    y: TokT[(BatchSize.type, BlockSize.type)]
  )

  def createBatch[D <: Dim.Dynamic](data: TokT[Tuple1[D]]) = {
    val length = data.sizeOf(dim[D])

    (BlockSize |<= length).map { ev =>
      import ev.given

      val ix = Tensor.randint(data.size(0) - BlockSize.size - 1)(BatchSize)
      val xb = ix.to(CPU).mapStack(idx => data(Take(BlockSize, drop = idx)))
      val yb = ix.to(CPU).mapStack(idx => data(Take(BlockSize, drop = idx + 1)))
      Batch(xb, yb)
    }.getOrElse {
      throw new IllegalArgumentException("too little training data")
    }
  }

  @main def run(): Unit = {
    org.bytedeco.pytorch.global.torch.manual_seed(42)

    val input = Source.fromFile("src/test/resources/tiny-shakespeare.txt").getLines.mkString("\n")
    object Tokenizer extends CharTokenizer(CharTokenizer.train[Tok.S](Seq(input)))
    val splitIdx = (input.size * 0.9).toInt
    class TrainingSize(size: Long) extends Dim.Dynamic(size)
    class ValidationSize(size: Long) extends Dim.Dynamic(size)
    val trainData = Tensor(Tokenizer.tokenize(input.take(splitIdx))).shaped[Tuple1[TrainingSize]]
    val valData = Tensor(Tokenizer.tokenize(input.drop(splitIdx))).shaped[Tuple1[ValidationSize]]

    for {
      ev1 <- BlockSize |<= trainData.sizeOf(TrainingSize(_))
      ev2 <- BlockSize |<= valData.sizeOf(ValidationSize(_))
    } {
      import ev1.given
      import ev2.given

      case object VocabSize extends Dim.Dynamic(Tokenizer.max.value + 1) // FIXME there really should be a dim/token thingy
      println(VocabSize) // 66
      case object MaxBlockSize extends Dim.Static[92L]
      case object DModel extends Dim.Static[32L]
      case object NHeads extends Dim.Static[4L]
      // val model = new Bigram(VocabSize)
      //val model = new SelfAttention(VocabSize, MaxBlockSize, DModel)
      //val model = new SelfAttention1(VocabSize, MaxBlockSize, DModel)
      val model = new SelfAttention2(VocabSize, MaxBlockSize, DModel, NHeads)

      def estimateLoss[D <: Dim.Dynamic](data: TokT[Tuple1[D]])(using BlockSize.type |<= D) =
        Tensor.noGrad {
          model.eval {
            val loss = Tensor.zeros(using device = Default(CPU))(64L).mapStack { _ =>
              val b = createBatch(data)
              model(b.x, b.y)
            }
            loss.mean
          }
        }

      val optimizer = AdamW(model.parameters, learningRate = learningRate)
      var finalLoss:Float = 0.0
      for (i <- 0.to(trainingRounds)) {
        scoped {
          val b = createBatch(trainData)
          val loss = model(b.x, b.y)
          optimizer.zeroGrad(setToNone = true)
          loss.backward()
          optimizer.step()
          finalLoss = loss.to(CPU).value

          if (i % printLossEvery == 0) {
            println(s"${i}: train: ${estimateLoss(trainData).to(CPU).value} val: ${estimateLoss(valData).to(CPU).value}")
          }
        }
      }

      println("final loss: " + finalLoss)

      class ExampleSize(size: Long) extends Dim.Dynamic(size)
      var gen = Tensor(Tokenizer.tokenize("\n")).shaped[Tuple1[ExampleSize]]
      whileDefined(gen.sizeOf(ExampleSize(_)) |<= MaxBlockSize) { ev =>
        import ev.given
        gen = model.extend(gen)
      }
      println(Tokenizer.untokenize(gen.to(CPU).value))

    }
  }
}
