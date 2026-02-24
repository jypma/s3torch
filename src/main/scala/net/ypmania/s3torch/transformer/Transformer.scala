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
import net.ypmania.s3torch.Shape.Select.Divided
import net.ypmania.s3torch.internal.Broadcast
import net.ypmania.s3torch.internal.Broadcastable
import net.ypmania.s3torch.nn.init

// Plain pytorch source: https://www.youtube.com/watch?v=ISNdQcPhsts
class Transformer[
  D <: Device,
  NHeads <: Long & Singleton,
  DModelN <: Long & Singleton,
  DModel <: Dim.Static[DModelN],
  BatchSize <: Dim,
  Dff <: Dim,
  T <: DType.Floaty]
  (dModel: DModel, batchSize: BatchSize, dff: Dff, nHeads: NHeads)
  (using Default[T], Default[D], DModelN |/ NHeads, RandomSource, ValueOf[NHeads], ValueOf[DModelN]) {

  type Tn[S <: Tuple] = Tensor[S, T, D]
  type Batch[SeqLen <: Dim] = Tn[(BatchSize, SeqLen, DModel)]

  class InputEmbeddings[VocabSize <: Dim](vocabSize: VocabSize) extends Module {
    val embedding = addModule("embedding", Embedding(vocabSize, dModel))

    def apply[S <: Shape](in: Tensor[S, Int32.type, D]): Tn[Append[S, DModel]] = embedding(in) * Math.sqrt(dModel.size.toDouble)
  }

  class PositionalEncoding[SeqLen <: Dim](seqLen: SeqLen, dropoutProb: Double) extends Module {
    val dropout = addModule("dropout", Dropout(dropoutProb))

    val position = Tensor.arangeOf(seqLen).unsqueezeAfter(Last)
    val indices = Tensor.arangeOf(dModel) /|/ 2
    val phase_offset = (Tensor.arangeOf(dModel) % 2) * (Math.PI * 0.5)
    val div_term = exp(indices * (-Math.log(10000.0) / dModel.size))
    val positionalEncodingDeltas = addBuffer("pe", sin(position * div_term + phase_offset))

    def apply(in: Batch[SeqLen]): Batch[SeqLen] = {
      dropout(in + positionalEncodingDeltas)
    }
  }

  class LayerNormalization(eps: Double = 1e-6) extends Module {
    val alpha = addParameter("alpha", Tensor.ones(1L))
    val bias = addParameter("bias", Tensor.zeros(1L))

    def apply[SeqLen <: Dim](in: Batch[SeqLen]): Batch[SeqLen] = {
      val mean = in.meanBy(Last)(using KeepDim) // FIXME verify this, video says "everything after batch" but picks last.
      val std = in.stdBy(Last)(using KeepDim)

      alpha * (in - mean) / (std + eps) + bias
    }
  }

  class FeedForward(dropoutProb: Double) extends Module {
    val l1 = addModule("l1", Linear(dModel, dff))
    val dropout = addModule("dropout", Dropout(dropoutProb))
    val l2 = addModule("l2", Linear(dff, dModel))

    def apply[SeqLen <: Dim](in: Batch[SeqLen]): Batch[SeqLen] = {
      in ~> l1.apply ~> relu ~> dropout.apply ~> l2.apply
    }
  }

  type AttentionScores[QSeqLen <: Dim, KVSeqLen <: Dim] = (BatchSize, Static[NHeads], QSeqLen, KVSeqLen)

  class MultiHeadAttention[QSeqLen <: Dim, KVSeqLen <: Dim](dropoutProb: Double) extends Module {
    type B = Batch[QSeqLen]

    val queryWeights = addModule("queryWeights", Linear(dModel, dModel)) // FIXME Maybe there should be no bias here if it's just a mul.
    val keyWeights = addModule("keyWeights", Linear(dModel, dModel))
    val valueWeights = addModule("valueWeights", Linear(dModel, dModel))
    val outputWeights = addModule("outputWeights", Linear(dModel, dModel))
    val dropout = addModule("dropout", Dropout(dropoutProb))

    /** Splits the dModel dimension into NHeads heads, and swap the SeqLen
      * and NHeads dimensions, so each head looks at a sequence of
      * vectors with that head's part of the original DModel. */
    private def splitHeads[SL <: Dim](b: Batch[SL]): Tn[(BatchSize, Static[NHeads], SL, DModel / NHeads)] =
      b.split[DModel].into[NHeads].transpose[SL, Static[NHeads]]

    private def joinHeads(h: Tn[(BatchSize, Static[NHeads], QSeqLen, DModel / NHeads)]) = {
      // TODO the original video needed a ".contiguous()" before the unsplit (.view) here, buta
      // we apparently don't need that...
      h.transpose[Static[NHeads], QSeqLen].unsplit[Divided[DModel]]
    }

    def apply(query: B, key: Batch[KVSeqLen], value: Batch[KVSeqLen]): Batch[QSeqLen] = apply[(QSeqLen, KVSeqLen)](query, key, value, None.asInstanceOf[Option[Tn[(QSeqLen, KVSeqLen)]]])

    def apply[M <: Tuple](query: B, key: Batch[KVSeqLen], value: Batch[KVSeqLen], mask: Option[Tn[M]])(using Broadcastable[AttentionScores[QSeqLen, KVSeqLen], M]): B = {
      val q = query ~> queryWeights.apply ~> splitHeads
      val k = key ~> keyWeights.apply ~> splitHeads
      val v = value ~> valueWeights.apply ~> splitHeads

      val attentionScores = (q `@` k.t / Math.sqrt(dModel.size.toDouble / summon[ValueOf[NHeads]].value))
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

  /** Applies an input through LayerNormalization, then into a
    * sub-layer, and then through a dropout, adding the result to the
    * original input. */
  class ResidualConnection(dropoutProb: Double) extends Module {
    val dropout = addModule("dropout", Dropout(dropoutProb))
    val norm = addModule("norm", new LayerNormalization)

    def apply[SeqLen <: Dim](sublayer: Batch[SeqLen] => Batch[SeqLen]): Batch[SeqLen] => Batch[SeqLen] =
      in => (in ~> norm.apply ~> sublayer ~> dropout.apply) + in
  }

  class EncoderBlock[SeqLen <: Dim](attention: MultiHeadAttention[SeqLen, SeqLen], feedForward: FeedForward, dropoutProb: Double) extends Module {
    addModule("attention", attention)
    addModule("feedForward", feedForward)
    val residual = addModules("residual", Seq.fill(2)(new ResidualConnection(dropoutProb)))

    def apply[M <: Tuple](mask: Tn[M])(in: Batch[SeqLen])(using Broadcastable[AttentionScores[SeqLen, SeqLen], M]): Batch[SeqLen] = {
      in
        ~> residual(0)(x => attention(x, x, x, Some(mask)))
        ~> residual(1)(feedForward.apply)
    }
  }

  class Encoder[SeqLen <: Dim](blocks: Seq[EncoderBlock[SeqLen]]) extends Module {
    addModules("blocks", blocks)
    val norm = addModule("norm", new LayerNormalization)

    def apply[M <: Tuple](mask: Tn[M])(in: Batch[SeqLen])(using Broadcastable[AttentionScores[SeqLen, SeqLen], M]): Batch[SeqLen] = {
      blocks.foldLeft(in)(_ ~> _(mask)) ~> norm.apply
    }
  }

  class DecoderBlock[SrcSeqLen <: Dim, TgtSeqLen <: Dim](
    selfAttention: MultiHeadAttention[TgtSeqLen, TgtSeqLen],
    crossAttention: MultiHeadAttention[TgtSeqLen, SrcSeqLen],
    feedForward: FeedForward,
    dropoutProb: Double) extends Module
  {
    addModule("selfAttention", selfAttention)
    addModule("crossAttention", crossAttention)
    addModule("feedForward", feedForward)
    val residual = addModules("residual", Seq.fill(3)(new ResidualConnection(dropoutProb)))

    def apply[EM <: Tuple, DM <: Tuple]
      (encoderOutput: Batch[SrcSeqLen], encoderMask: Tn[EM], decoderMask: Tn[DM])(in: Batch[TgtSeqLen])
      (using Broadcastable[AttentionScores[TgtSeqLen, TgtSeqLen], DM], Broadcastable[AttentionScores[TgtSeqLen, SrcSeqLen], EM])
        : Batch[TgtSeqLen] = {
      in
        ~> residual(0)(x => selfAttention(x, x, x, Some(decoderMask)))
        ~> residual(1)(x => crossAttention(x, encoderOutput, encoderOutput, Some(encoderMask)))
        ~> residual(2)(feedForward.apply)
    }
  }

  class Decoder[SrcSeqLen <: Dim, TgtSeqLen <: Dim](blocks: Seq[DecoderBlock[SrcSeqLen, TgtSeqLen]]) extends Module {
    addModules("blocks", blocks)
    val norm = addModule("norm", new LayerNormalization)

    def apply[EM <: Tuple, DM <: Tuple]
      (encoderOutput: Batch[SrcSeqLen], encoderMask: Tn[EM], decoderMask: Tn[DM])(in: Batch[TgtSeqLen])
      (using Broadcastable[AttentionScores[TgtSeqLen, TgtSeqLen], DM], Broadcastable[AttentionScores[TgtSeqLen, SrcSeqLen], EM])
        : Batch[TgtSeqLen] = {
//    def apply[M <: Tuple](encoderOutput: Batch[SeqLen], encoderMask: Tn[M], decoderMask: Tn[M])(in: Batch[SeqLen])(using Broadcastable[AttentionScores[SeqLen], M]): Batch[SeqLen] = {
      blocks.foldLeft(in)(_ ~> _(encoderOutput, encoderMask, decoderMask)) ~> norm.apply
    }
  }

  class Projection[VocabSize <: Dim](vocabSize: VocabSize) extends Module {
    val proj = addModule("proj", Linear(dModel, vocabSize))

    def apply[SeqLen <: Dim](in: Batch[SeqLen]): Tn[(BatchSize, SeqLen, VocabSize)] = {
      proj(in).log_softmax[VocabSize]
    }
  }

  class Main[
    SrcSeqLen <: Dim,
    TgtSeqLen <: Dim,
    SrcVocabSize <: Dim,
    TgtVocabSize <: Dim
  ](
    encoder: Encoder[SrcSeqLen],
    decoder: Decoder[SrcSeqLen, TgtSeqLen],
    sourceEmb: InputEmbeddings[SrcVocabSize],
    targetEmb: InputEmbeddings[TgtVocabSize],
    sourcePos: PositionalEncoding[SrcSeqLen],
    targetPos: PositionalEncoding[TgtSeqLen],
    projection: Projection[TgtVocabSize]
  ) extends Module {
    addModule("encoder", encoder)
    addModule("decoder", decoder)
    addModule("sourceEmb", sourceEmb)
    addModule("targetEmb", targetEmb)
    addModule("sourcePos", sourcePos)
    addModule("targetPos", targetPos)
    addModule("projection", projection)

    parameters.foreach(init.xavier_uniform)

    def encode[M <: Tuple](src: Tensor[(BatchSize, SrcSeqLen), Int32.type, D], srcMask: Tn[M])(using Broadcastable[AttentionScores[SrcSeqLen, SrcSeqLen], M]): Batch[SrcSeqLen] = {
      src ~> sourceEmb.apply ~> sourcePos.apply ~> encoder(srcMask)
    }

    def decode[EM <: Tuple, DM <: Tuple]
      (encoderOutput: Batch[SrcSeqLen], encoderMask: Tn[EM], decoderMask: Tn[DM])(tgt: Tensor[(BatchSize, TgtSeqLen), Int32.type, D])
      (using Broadcastable[AttentionScores[TgtSeqLen, TgtSeqLen], DM], Broadcastable[AttentionScores[TgtSeqLen, SrcSeqLen], EM])
        : Batch[TgtSeqLen] = {
      tgt ~> targetEmb.apply ~> targetPos.apply ~> decoder(encoderOutput, encoderMask, decoderMask)
    }

    def project(x: Batch[TgtSeqLen]): Tn[(BatchSize, TgtSeqLen, TgtVocabSize)] = projection(x)
  }
}

object Transformer {
  def apply[
    D <: Device,
    T <: DType.Floaty,
    SrcVocabSize <: Dim,
    TgtVocabSize <: Dim,
    SrcSeqLen <: Dim,
    TgtSeqLen <: Dim,
    DModelN <: Long & Singleton,
    DModel <: Dim.Static[DModelN],
    NHeads <: Long & Singleton,
    DFF <: Dim,
    BatchSize <: Dim
  ](
    batchSize: BatchSize,
    srcVocabSize: SrcVocabSize,
    tgtVocabSize: TgtVocabSize,
    srcSeqLen: SrcSeqLen,
    tgtSeqLen: TgtSeqLen,
    dModel: DModel,
    /** Size of the hidden feed-forward layer, default to 2048 */
    dFF: DFF,
    /** Number of attention heads (H), default to 8 */
    nHeads: NHeads,
    /** Number of encoder and decoder layers (N), default to 6 */
    coderLayers: Int,
    dropoutProb: Double // default to 0.1
  )(using
    Default[T], Default[D], DModelN |/ NHeads, RandomSource, ValueOf[NHeads], ValueOf[DModelN]
  ) = {
    val t = new Transformer[D, NHeads, DModelN, DModel, BatchSize, DFF, T](dModel, batchSize, dFF, nHeads)
    val srcEmbed = new t.InputEmbeddings(srcVocabSize)
    val tgtEmbed = new t.InputEmbeddings(tgtVocabSize)

    val srcPos = new t.PositionalEncoding(srcSeqLen, dropoutProb)
    val tgtPos = new t.PositionalEncoding(tgtSeqLen, dropoutProb)

    val encoder = new t.Encoder(
      0.until(coderLayers).map { i =>
        new t.EncoderBlock(
          new t.MultiHeadAttention[SrcSeqLen, SrcSeqLen](dropoutProb),
          new t.FeedForward(dropoutProb),
          dropoutProb
        )
      }
    )

    val decoder = new t.Decoder(
      0.until(coderLayers).map { i =>
        new t.DecoderBlock(
          new t.MultiHeadAttention[TgtSeqLen, TgtSeqLen](dropoutProb),
          new t.MultiHeadAttention[TgtSeqLen, SrcSeqLen](dropoutProb),
          new t.FeedForward(dropoutProb),
          dropoutProb
        )
      }
    )

    val projection = new t.Projection(tgtVocabSize)
    t.Main(encoder, decoder, srcEmbed, tgtEmbed, srcPos, tgtPos, projection)
  }
}
