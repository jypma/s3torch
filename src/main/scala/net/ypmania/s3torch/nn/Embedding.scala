package net.ypmania.s3torch.nn

import net.ypmania.s3torch._
import org.bytedeco.pytorch

import DType._
import Tuple.:*

class Embedding[D <: Device, OutT <: DType, Num <: Dim, Emb <: Dim] (native: pytorch.EmbeddingImpl) extends AbstractModule[D, OutT](native) {
  type This[D <: Device, T <: DType] = Embedding[D, T, Num, Emb]

  def apply[S <: Shape, T <: (Int64.type | Int32.type)](in: Tensor[S, T, D]): Tensor[S :* Emb, OutT, D] =
    new Tensor(native.forward(in.native))
}

object Embedding {
    def apply[Num <: Dim, Emb <: Dim, D <: Device, T <: DType.Floaty](numEmbeddings: Num, embeddingDim: Emb)(using rnd: RandomSource, t: Default[T], d: Default[D]): Embedding[D, T, Num, Emb] = rnd(
      new Embedding(
        new pytorch.EmbeddingImpl(
          new pytorch.EmbeddingOptions(numEmbeddings.size, embeddingDim.size)
        )
      ).toDeviceDType
    )
}
