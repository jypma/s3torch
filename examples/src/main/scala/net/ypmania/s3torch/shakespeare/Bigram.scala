package net.ypmania.s3torch.shakespeare

import net.ypmania.s3torch.Batched
import net.ypmania.s3torch.Batched1
import net.ypmania.s3torch.nn.Module
import net.ypmania.s3torch.nn.Embedding
import net.ypmania.s3torch.Index
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

class Bigram[VocabSize <: Dim, D <: Device](vocabSize: VocabSize)(using Default[D]) extends Module {
  val embedding = addModule("embedding", Embedding(vocabSize, vocabSize))

  def apply[B <: Dim, L <: Dim, T <: Int64](idx: Tensor[(B, L), T, D], targets: Tensor[(B, L), T, D]): Tensor[EmptyTuple.type, Float32, D] = {
    val expected = embedding(idx).view.merge[L]
    val actual = targets.view.merge[L]
    CrossEntropy(expected, actual)
  }

  def extend[S <: Tuple, B <: Tuple, L <: Dim.Dynamic, T <: Int64](idx: Tensor[S, T, D])(using b: Batched1[B, L, S]): Tensor[S, T, D] = {
    import b.given

    val expected = embedding(idx)(dim[L] % Index.Last)
    val probs = expected.softmax(vocabSize)
    val next = probs.multinomial(Dim.One).toDTypeOf(idx)

    idx.cat(next)(dim[L])
  }
}
