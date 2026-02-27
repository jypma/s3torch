package net.ypmania.s3torch.nn

import net.ypmania.s3torch.*

import Dim.Static
import scala.reflect.ClassTag
import DType.*
import Device.CPU
import net.ypmania.s3torch.Device.CUDA

class EmbeddingSpec extends UnitSpec {
  case object NumEmbeddings extends Static[10L]
  case object EmbeddingDim extends Static[4L]
  case object BatchSize extends Static[32L]

  describe("Embedding") {
    it("can apply to a simple vector") {
      val emb = Embedding(NumEmbeddings, EmbeddingDim)
      val in = Tensor((1, 2, 1))
      val res = emb(in)
      val resType: Tensor[(Static[3L], EmbeddingDim.type), Float32, CPU.type] = res
      assert(res.value.toSeq === Seq(
        Seq(0.8487, 0.6920, -0.3160, -2.1152),
        Seq(0.3222, -1.2633, 0.3499, 0.3081),
        Seq(0.8487, 0.6920, -0.3160, -2.1152)
      ))
    }

    it("can change to a different DType") {
      val emb = Embedding(NumEmbeddings, EmbeddingDim).to(float64)
      val in = Tensor((1, 2, 1))
      val res = emb(in)
      val resType: Tensor[(Static[3L], EmbeddingDim.type), Float64, CPU.type] = res
    }

    it("can apply to a 2D batch") {
      val emb = Embedding(NumEmbeddings, EmbeddingDim)
      val in = Tensor((((1, 2, 3)), ((4,5,6))))
      val res = emb(in)
      val resType: Tensor[(Static[2L], Static[3L], EmbeddingDim.type), Float32, CPU.type] = res
    }

    it("can run on the GPU") {
      val emb = Embedding(NumEmbeddings, EmbeddingDim).to(CUDA)
      val in = Tensor((1, 2, 1)).to(CUDA)
      // Can't emb(in).to(CPU) here
      // See https://github.com/scala/scala3/issues/25204
      val res = emb(in)
      val resType: Tensor[(Static[3L], EmbeddingDim.type), Float32, CUDA.type] = res
      val r = res.to(CPU)
      // Same values as CPU case above
      assert(r.value.toSeq === Seq(
        Seq(0.8487, 0.6920, -0.3160, -2.1152),
        Seq(0.3222, -1.2633, 0.3499, 0.3081),
        Seq(0.8487, 0.6920, -0.3160, -2.1152)
      ))
    }
  }
}
