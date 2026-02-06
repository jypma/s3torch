package net.ypmania.s3torch.nn

import org.bytedeco.pytorch

import net.ypmania.s3torch.*
import DType.*

import Tuple.:*
import AbstractModule.CreationDType

class Embedding[OutT <: DType, Num <: Dim, Emb <: Dim] (native: pytorch.EmbeddingImpl) extends AbstractModule(native) {
  type This[T <: DType] = Embedding[T, Num, Emb]

  def apply[S <: Shape, T <: (Int64.type | Int32.type)](in: Tensor[S,T]): Tensor[S :* Emb, OutT] =
    new Tensor(native.forward(in.native))
}

object Embedding {
    def apply[Num <: Dim, Emb <: Dim, T <: DType.Floaty](numEmbeddings: Num, embeddingDim: Emb)(using rnd: RandomSource, t: Default[T]): Embedding[T, Num, Emb] = rnd(
      new Embedding(
        new pytorch.EmbeddingImpl(
          new pytorch.EmbeddingOptions(numEmbeddings.size, embeddingDim.size)
        )
      ).toDType
    )
}
