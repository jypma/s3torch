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
    val indices = Tensor.arangeOf(dModel).floor_divide(2)
    val phase_offset = Tensor.arangeOf(dModel).remainder(2) * (Math.PI * 0.5)
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
      l2(
        dropout(
          relu(l1(in))
        )
      )
    }
  }

  class MultiHeadAttention(dropoutProb: Double) extends Module {
    val queryWeights = addModule("queryWeights", Linear(dModel, dModel)) // FIXME Maybe there should be no bias here if it's just a mul.
    val keyWeights = addModule("keyWeights", Linear(dModel, dModel))
    val valueWeights = addModule("valueWeights", Linear(dModel, dModel))
    val outputWeights = addModule("outputWeights", Linear(dModel, dModel))
    val dropout = addModule("dropout", Dropout(dropoutProb))

    def apply(query: Batch, key: Batch, value: Batch, mask: Batch): Batch = {
      val q = queryWeights(query)
      val k = keyWeights(key)
      val v = valueWeights(value)

      // Split the dModel dimension into NHeads heads
      val s = q.split[DModel].into[NHeads]
      val sType: Tensor[(BatchSize, SeqLen, Static[NHeads], DModel / NHeads), T] = s

      // Just a temp test that this keeps compiling
      val tstUnsplit = s.unsplit[Divided[DModel]]
      val tstUnsplitT: Tensor[(BatchSize, SeqLen, DModel), T] = tstUnsplit

      // Swap the SeqLen and NHeads dimensions
      val st = s.transpose[SeqLen, Static[NHeads]]

      val stType: Tensor[(BatchSize, Static[NHeads], SeqLen, DModel / NHeads), T] = st

      ???
    }
  }
}
