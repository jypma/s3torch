package net.ypmania.s3torch.nn

import org.scalatest.Assertions._

import net.ypmania.s3torch.*

import Dim.Static
import Dim.Dynamic
import scala.reflect.ClassTag
import net.ypmania.s3torch.internal.Torch
import DType.*

class EmbeddingSpec extends UnitSpec {
  case object NumEmbeddings extends Static[10L]
  case object EmbeddingDim extends Static[4L]
  case object BatchSize extends Static[32L]

  describe("Embedding") {
    it("can apply to a simple vector") {
      val emb = Embedding(NumEmbeddings, EmbeddingDim)
      val in = Tensor((1, 2, 1))
      val res = emb(in)
      val resType: Tensor[(Static[3L], EmbeddingDim.type), Float32.type] = res
    }

    it("can apply to a 2D batch") {
      val emb = Embedding(NumEmbeddings, EmbeddingDim)
      val in = Tensor((((1, 2, 3)), ((4,5,6))))
      val res = emb(in)
      val resType: Tensor[(Static[2L], Static[3L], EmbeddingDim.type), Float32.type] = res
     }
  }
}
