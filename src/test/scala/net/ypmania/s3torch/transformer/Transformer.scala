package net.ypmania.s3torch.transformer

import net.ypmania.s3torch.nn.Module
import net.ypmania.s3torch.*
import net.ypmania.s3torch.DType.*
import Tensor._
import net.ypmania.s3torch.Default
import net.ypmania.s3torch.Dim.Static
import net.ypmania.s3torch.Shape.Select.Last
import scala.Tuple.Append
import net.ypmania.s3torch.internal.ReduceOperand
import net.ypmania.s3torch.nn.Dropout
import net.ypmania.s3torch.nn.Embedding
import net.ypmania.s3torch.nn.Linear
import net.ypmania.s3torch.Dim.*
import net.ypmania.s3torch.Shape.Select.Idx
import net.ypmania.s3torch.Shape.Select.At
import net.ypmania.s3torch.Shape.Select.Divided
import net.ypmania.s3torch.internal.Broadcast
import net.ypmania.s3torch.internal.Broadcastable

// Plain pytorch source: https://www.youtube.com/watch?v=ISNdQcPhsts
class Transformer[
  NHeads <: Long & Singleton,
  DModelN <: Long & Singleton,
  DModel <: Dim.Static[DModelN],
  VocabSize <: Dim,
  SeqLen <: Dim,
  BatchSize <: Dim,
  T <: DType.Floaty](dModel: DModel, vocabSize: VocabSize, seqLen: SeqLen, batchSize: BatchSize, nHeads: NHeads)(using Default[T], DModelN |/ NHeads, RandomSource, ValueOf[NHeads], ValueOf[DModelN]) {
  type Batch = Tensor[(BatchSize, SeqLen, DModel), T]

  class InputEmbeddings extends Module {
    val embedding = addModule("embedding", Embedding(vocabSize, dModel))

    def apply[S <: Shape](in: Tensor[S, Int32.type]): Tensor[Append[S, DModel], T] = embedding(in) * Math.sqrt(dModel.size.toDouble)
  }

  class PositionalEncoding(dropoutProb: Double) extends Module {
    val dropout = addModule("dropout", Dropout(dropoutProb))

    val position = Tensor.arangeOf(seqLen).unsqueezeAfter(Last)
    val indices = Tensor.arangeOf(dModel) /|/ 2
    val phase_offset = (Tensor.arangeOf(dModel) % 2) * (Math.PI * 0.5)
    val div_term = exp(indices * (-Math.log(10000.0) / dModel.size))
    val positionalEncodingDeltas = addBuffer("pe", sin(position * div_term + phase_offset))

    def apply(in: Batch): Batch = {
      dropout(in + positionalEncodingDeltas)
    }
  }

  class LayerNormalization(eps: Double = 1e-6) extends Module {
    val alpha = addParameter("alpha", Tensor.ones(1L))
    val bias = addParameter("bias", Tensor.zeros(1L))

    def apply(in: Batch): Batch = {
      val mean = in.meanBy(Last)(using KeepDim) // FIXME verify this, video says "everything after batch" but picks last.
      val std = in.stdBy(Last)(using KeepDim)

      alpha * (in - mean) / (std + eps) + bias
    }
  }

  class FeedForward[Dff <: Dim, SeqLen <: Dim](dff: Dff, seqLen: SeqLen, dropoutProb: Double) extends Module {
    val l1 = addModule("l1", Linear(dModel, dff))
    val dropout = addModule("dropout", Dropout(dropoutProb))
    val l2 = addModule("l2", Linear(dff, dModel))

    def apply(in: Batch): Batch = {
      in ~> l1.apply ~> relu ~> dropout.apply ~> l2.apply
    }
  }

  type AttentionScores = (BatchSize, Static[NHeads], SeqLen, SeqLen)

  class MultiHeadAttention(dropoutProb: Double) extends Module {
    val queryWeights = addModule("queryWeights", Linear(dModel, dModel)) // FIXME Maybe there should be no bias here if it's just a mul.
    val keyWeights = addModule("keyWeights", Linear(dModel, dModel))
    val valueWeights = addModule("valueWeights", Linear(dModel, dModel))
    val outputWeights = addModule("outputWeights", Linear(dModel, dModel))
    val dropout = addModule("dropout", Dropout(dropoutProb))

    /** Splits the dModel dimension into NHeads heads, and swap the SeqLen
      * and NHeads dimensions, so each head looks at a sequence of
      * vectors with that head's part of the original DModel. */
    private def splitHeads(b: Batch): Tensor[(BatchSize, Static[NHeads], SeqLen, DModel / NHeads), T] =
      b.split[DModel].into[NHeads].transpose[SeqLen, Static[NHeads]]

    private def joinHeads(h: Tensor[(BatchSize, Static[NHeads], SeqLen, DModel / NHeads), T]) = {
      // FIXME the original video needed a ".contiguous()" before the unsplit (.view) here, but
      // we apparently don't need that...
      h.transpose[Static[NHeads], SeqLen].unsplit[Divided[DModel]]
    }

    def apply(query: Batch, key: Batch, value: Batch): Batch = apply[(SeqLen, SeqLen)](query, key, value, None.asInstanceOf[Option[Tensor[(SeqLen, SeqLen), T]]])

    def apply[M <: Tuple](query: Batch, key: Batch, value: Batch, mask: Option[Tensor[M, T]])(using Broadcastable[AttentionScores, M]): Batch = {
      val q = query ~> queryWeights.apply ~> splitHeads
      val k = key ~> keyWeights.apply ~> splitHeads
      val v = value ~> valueWeights.apply ~> splitHeads

      val attentionScores = (q `@` k.t / Math.sqrt(dModel.size.toDouble / nHeads))
        .when(mask.map(_ #== 0))(_.maskedFilled(_, 1e-9))
        .softmax(Last)
        ~> dropout.apply

      // TODO save the attention scores somehow, they're apparently needed for visualization later.
      attentionScores
        `@` v
        ~> joinHeads.apply
        ~> outputWeights.apply
    }
  }
}
