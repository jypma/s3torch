package net.ypmania.s3torch.shakespeare

import net.ypmania.s3torch.Batched
import net.ypmania.s3torch.Batched1
import net.ypmania.s3torch.nn.Module
import net.ypmania.s3torch.nn.Embedding
import net.ypmania.s3torch.Index
import net.ypmania.s3torch.Index.Last
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Dim.|<=
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
import net.ypmania.s3torch.Index.Take
import scala.Tuple.:*

// DModel (number of embedding dimensions): 32
// Note for presentation: Video timestamp 1:21:13
class SelfAttention1[VocabSize <: Dim, MaxBlockSize <: Dim, DModel <: Dim, D <: Device](vocabSize: VocabSize, maxBlockSize: MaxBlockSize, dModel: DModel)(using Default[D]) extends Module {
  class Head[Size <: Dim](size: Size) extends Module {
    val key = addModule("key", Linear(dModel, size, bias = false))
    val query = addModule("query", Linear(dModel, size, bias = false))
    val value = addModule("value", Linear(dModel, size, bias = false))
    val tril = addBuffer("tril", Tensor.ones(maxBlockSize, maxBlockSize).tril())

    def apply[S <: Tuple, B <: Tuple, L <: Dim](x: Tensor[S, Float32, D])(using b: Batched[B, (L, DModel), S])(using L |<= MaxBlockSize) = {
      import b.given

      val k = key(x)
      val q = query(x)
      // Let's take only the part of tril up to [L]
      val trilPart = tril(Take(x.sizeOf(dim[L])), Take(x.sizeOf(dim[L])))
      val wei =
        ((q `@` k.t) / Math.sqrt(dModel.size.toDouble))
          .maskedFilled(trilPart #== 0.0, Double.NegativeInfinity) //  1e-20 works a LOT worse here.
          .softmax(Last)

      val v = value(x)
      wei `@` v
    }
  }

  val tokenEmbedding = addModule("tokenEmbedding", Embedding(vocabSize, dModel))
  val positionEmbedding = addModule("positionEmbedding", Embedding(maxBlockSize, dModel))
  val lmHead = addModule("lmHead", Linear(dModel, vocabSize))
  val saHead = addModule("saHead", new Head(dModel))

  def apply[B <: Dim, Length <: Dim, T <: Int64](idx: Tensor[(B, Length), T, D], targets: Tensor[(B, Length), T, D])(using Length |<= MaxBlockSize): Tensor[EmptyTuple.type, Float32, D] = {
    val predicted = logits(idx).view.merge[Length]
    val actual = targets.view.merge[Length]
    CrossEntropy(predicted, actual)
  }

  // TODO rewrite, this actually extends Length with one, so |<= isn't guaranteed.
  def extend[S <: Tuple, B <: Tuple, Length <: Dim.Dynamic, T <: Int64](idx: Tensor[S, T, D])(using b: Batched1[B, Length, S])(using Length |<= MaxBlockSize): Tensor[S, T, D] = {
    import b.given

    val predicted = logits(idx)(dim[Length] % Last)
    val probs = predicted.softmax(vocabSize)
    val next = probs.multinomial(Dim.One).toDTypeOf(idx)

    idx.cat(next)(dim[Length])
  }

  private def logits[S <: Tuple, B <: Tuple, Length <: Dim, T <: Int64](idx: Tensor[S, T, D])(using b: Batched1[B, Length, S])(using Length |<= MaxBlockSize): Tensor[B :* Length :* VocabSize, Float32, D] = {
    import b.given

    val tok = tokenEmbedding(idx)
    val len = idx.sizeOf(dim[Length])
    val pos = positionEmbedding(Tensor.arangeOfD(len))
    val r = tok + pos
    lmHead(saHead(r))
  }
}

object SelfAttention1 {
}
