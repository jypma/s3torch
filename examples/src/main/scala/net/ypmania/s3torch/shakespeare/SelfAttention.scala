package net.ypmania.s3torch.shakespeare

import net.ypmania.s3torch.Batched
import net.ypmania.s3torch.Batched1
import net.ypmania.s3torch.nn.Module
import net.ypmania.s3torch.nn.Embedding
import net.ypmania.s3torch.Index
import net.ypmania.s3torch.Index.Last
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Select
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.DType.Int64
import net.ypmania.s3torch.Default
import net.ypmania.s3torch.nn.CrossEntropy
import net.ypmania.s3torch.DType.Float32
import net.ypmania.s3torch.Select.dim
import net.ypmania.s3torch.internal.Multinomial
import net.ypmania.s3torch.nn.Linear
import scala.Tuple.:*

// DModel (number of embedding dimensions): 32
// Video position: 1:01:47
class SelfAttention[VocabSize <: Dim, MaxBlockSize <: Dim, DModel <: Dim, D <: Device](vocabSize: VocabSize, maxBlockSize: MaxBlockSize, dModel: DModel)(using Default[D]) extends Module {
  val tokenEmbedding = addModule("tokenEmbedding", Embedding(vocabSize, dModel))
  val positionEmbedding = addModule("positionEmbedding", Embedding(maxBlockSize, dModel))
  val lmHead = addModule("lmHead", Linear(dModel, vocabSize))

  def apply[B <: Dim, Length <: Dim, T <: Int64](idx: Tensor[(B, Length), T, D], targets: Tensor[(B, Length), T, D]): Tensor[EmptyTuple.type, Float32, D] = {
    val predicted = logits(idx).view.merge[Length]
    val actual = targets.view.merge[Length]
    CrossEntropy(predicted, actual)
  }

  def extend[S <: Tuple, B <: Tuple, Length <: Dim.Dynamic, T <: Int64](idx: Tensor[S, T, D])(using b: Batched1[B, Length, S]): Tensor[S, T, D] = {
    import b.given

    val predicted = logits(idx)(dim[Length] % Last)
    val probs = predicted.softmax(vocabSize)
    val next = probs.multinomial(Dim.One).toDTypeOf(idx)

    idx.cat(next)(dim[Length])
  }

  private def logits[S <: Tuple, B <: Tuple, Length <: Dim, T <: Int64](idx: Tensor[S, T, D])(using b: Batched1[B, Length, S]): Tensor[B :* Length :* VocabSize, Float32, D] = {
    import b.given

    val tok = tokenEmbedding(idx)
    val len = idx.sizeOf(dim[Length])
    val pos = positionEmbedding(Tensor.arangeOfD(len))
    val r = tok + pos
    lmHead(r)
  }


}
