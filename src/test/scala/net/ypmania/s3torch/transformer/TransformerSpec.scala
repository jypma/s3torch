package net.ypmania.s3torch.transformer

import org.scalatest.Assertions._
import net.ypmania.s3torch.UnitSpec
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Dim.Static
import net.ypmania.s3torch.DType.*
import net.ypmania.s3torch.Default
import net.ypmania.s3torch.Dim.DivisibleBy

class TransformerSpec extends UnitSpec {
  case object DModel extends Dim.Static[4L]
  case object VocabSize extends Dim.Static[3L]
  case object SeqLen extends Dim.Static[3L]
  case object Dff extends Dim.Static[5L]
  case object BatchSize extends Dim.Static[1L]

  val transformer = new Transformer(DModel, VocabSize, SeqLen, BatchSize, 4L)

  describe("Transformer.InputEmbedding") {
    val inputEmb = new transformer.InputEmbeddings

    it("should apply a scaled embedding") {
      val in = Tensor((0, 1, 2, 1)) // a test value for each element in our VocabSize
      val res = inputEmb(in)
      val resType = res

      assert(res.value === List(
        Seq(3.0819, -0.5868, -4.3575, 1.1368),
        Seq(-2.1690, -2.7971, 0.8066, 1.6760),
        Seq(-1.4385, -0.8066, -1.1932, 0.3640),
        Seq(-2.1690, -2.7971, 0.8066, 1.6760))
      )
    }
  }

  describe("Transformer.PositionalEncoding") {
    val posEnc = new transformer.PositionalEncoding(0.0)

    it("should add positional encoding deltas to an input") {
      val in = Tensor.zeros(BatchSize, SeqLen, DModel)
      val res = posEnc(in)

      assert(res.value === Seq(Seq(Seq(0.0, 1.0), Seq(0.8414, 0.5403), Seq(0.9092, -0.4161))))
    }
  }

  describe("LayerNormalization") {
    val norm = new transformer.LayerNormalization
    it("should normalize by mean and std") {
      val in = Tensor.zeros(BatchSize, SeqLen, DModel)
      in((0, 0, 0)) = 1.0
      in((0, 0, 1)) = 2.0
      in((0, 0, 2)) = 3.0
      val res = norm(in)
      assert(res.value === Seq(
        Seq(
          Seq(-0.38729805, 0.38729805, 1.1618942, -1.1618942),
          Seq(0.0, 0.0, 0.0, 0.0),
          Seq(0.0, 0.0, 0.0, 0.0)
        )
      ))
    }
  }

  describe("FeedForward") {
    val ff = new transformer.FeedForward(Dff, SeqLen, 0.0)
    it("should run through two layers") {
      val in = Tensor.zeros(BatchSize, SeqLen, DModel)
      in((0, 1, 0)) = 1.0
      val res = ff(in)

      assert(res.value === Seq(
        Seq(
          Seq(0.1382, 0.3746, -0.2638, 0.2976),
          Seq(0.1950, 0.4045, -0.1572, 0.3541),
          Seq(0.1382, 0.3746, -0.2638, 0.2976)
        )
      ))
    }
  }

  describe("MultiHeadAttention") {
    val mh = new transformer.MultiHeadAttention(0.5)

    it("should apply key and value to the batch") {
      val in = Tensor.zeros(BatchSize, SeqLen, DModel)
      val key = Tensor.zeros(BatchSize, SeqLen, DModel)
      val value = Tensor.zeros(BatchSize, SeqLen, DModel)
      val res = mh(in, key, value)

      assert(res.value === Seq(
        Seq(
          Seq(0.5045, 0.2997, -0.2170, -0.4354),
          Seq(0.4513, 0.2983, -0.2376, -0.3063),
          Seq(0.5574, 0.2168, -0.2085, -0.5011)
        )
      ))
    }
  }
}
