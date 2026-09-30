package net.ypmania.s3torch.nn

import net.ypmania.s3torch.UnitSpec
import net.ypmania.s3torch.Tensor

class LayerNormSpec extends UnitSpec{
  describe("LayerNorm with bias") {
    val layerNorm = LayerNorm(2L, 2L)

    it("should normalize a tensor with the same size") {
      val in = Tensor.zeros(2L, 2L)
      in((0,0)) = 1
      val out = layerNorm(in)
      assert(out.size == in.size)
      assert(out.value === Seq(
        Seq(1.7320, -0.5773),
        Seq(-0.5773, -0.5773)
      ))
    }

    it("should normalize a tensor with a batch dimension") {
      val in = Tensor.zeros(2L, 2L, 2L)
      in((0,0,0)) = 1
      val out = layerNorm(in)
      assert(out.size == in.size)
      assert(out.value === Seq(
        Seq(
          Seq(1.7320, -0.5773),
          Seq(-0.5773, -0.5773)
        ), Seq(
          Seq(0.0, 0.0),
          Seq(0.0, 0.0)
        )
      ))
    }

    it("should not copmile if using layerNorm with an incompatible tensor") {
      val in = Tensor.zeros(2L, 3L, 5L)
      in((0,0,0)) = 1
      //val out = layerNorm(in) // does not compile
    }
  }
}
