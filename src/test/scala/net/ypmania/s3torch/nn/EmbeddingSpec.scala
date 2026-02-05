package net.ypmania.s3torch.nn

import org.scalatest.Assertions._

import net.ypmania.s3torch.*

import Dim.Static
import Dim.Dynamic
import scala.reflect.ClassTag
import net.ypmania.s3torch.internal.Torch
import DType.*
import Device.CPU
import net.ypmania.s3torch.Device.CUDA
import net.ypmania.s3torch.Device.CPU

class EmbeddingSpec extends UnitSpec {
  case object NumEmbeddings extends Static[10L]
  case object EmbeddingDim extends Static[4L]
  case object BatchSize extends Static[32L]

  describe("Embedding") {
    it("can apply to a simple vector") {
      val emb = Embedding(NumEmbeddings, EmbeddingDim)
      val in = Tensor((1, 2, 1))
      val res = emb(in)
      val resType: Tensor[(Static[3L], EmbeddingDim.type), Float32.type, CPU.type] = res
      println(res.value.toSeq)
    }

    it("can change to a different DType") {
      val emb = Embedding(NumEmbeddings, EmbeddingDim).to(Float64)
      val in = Tensor((1, 2, 1))
      val res = emb(in)
      val resType: Tensor[(Static[3L], EmbeddingDim.type), Float64.type, CPU.type] = res
    }

    it("can apply to a 2D batch") {
      val emb = Embedding(NumEmbeddings, EmbeddingDim)
      val in = Tensor((((1, 2, 3)), ((4,5,6))))
      val res = emb(in)
      val resType: Tensor[(Static[2L], Static[3L], EmbeddingDim.type), Float32.type, CPU.type] = res
    }

    it("can run on the GPU") {
      val emb = Embedding(NumEmbeddings, EmbeddingDim).to(CUDA)
      val in = Tensor((1, 2, 1)).to(CUDA)

      // This seems to compile fine:
      val res = emb(in)
      val r = res.to(CPU)

      // Yet. Scala 3.8.1 crashes on this line:
      // java.lang.AssertionError: assertion failed: no owner from  <none>/ <none> in emb.apply[net.ypmania.s3torch.Dim.Static[(3L : Long)] *: EmptyTuple,
      //   net.ypmania.s3torch.DType.Int32.type](in).<none>
      val r2 = emb(in).to(CPU)
    }
  }
}
