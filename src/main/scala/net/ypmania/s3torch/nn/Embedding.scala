package net.ypmania.s3torch.nn

import org.bytedeco.pytorch

import net.ypmania.s3torch.*
import DType.*

import Tuple.:*

class Embedding[OutT <: DType, Num <: Dim, Emb <: Dim] (native: pytorch.EmbeddingImpl) extends AbstractModule(native) {
  def apply[S <: Shape, T <: (Int64.type | Int32.type)](in: Tensor[S,T]): Tensor[S :* Emb, OutT] =
    new Tensor(native.forward(in.native))
}

object Embedding {
  /** Creates a new embedding layer, with the inner parameters created as [T] */
  def apply[T <: DType](using dtype: Default[T], rnd:RandomSource) = new Apply(dtype.value)

  class Apply[T <: DType](dtype: T)(using rnd:RandomSource) {
    def apply[Num <: Dim, Emb <: Dim](numEmbeddings: Num, embeddingDim: Emb): Embedding[T, Num, Emb] = rnd(
      new Embedding(
        new pytorch.EmbeddingImpl(
          new pytorch.EmbeddingOptions(numEmbeddings.size, embeddingDim.size)
        )
      ).to(dtype)
    )
  }
}
