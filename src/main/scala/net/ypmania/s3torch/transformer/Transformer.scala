package net.ypmania.s3torch.transformer

import net.ypmania.s3torch.DType._
import net.ypmania.s3torch.Dim._
import net.ypmania.s3torch.Shape.Select.Last
import net.ypmania.s3torch._
import net.ypmania.s3torch.internal.Broadcast
import net.ypmania.s3torch.internal.Broadcastable
import net.ypmania.s3torch.nn.Dropout
import net.ypmania.s3torch.nn.Embedding
import net.ypmania.s3torch.nn.Linear
import net.ypmania.s3torch.nn.Module
import net.ypmania.s3torch.Batched
import net.ypmania.s3torch.nn.init

import scala.Tuple.Append

import Tensor._
import Tuple._

// Plain pytorch source: https://www.youtube.com/watch?v=ISNdQcPhsts
class Transformer[
  D <: Device,
  NHeads <: Dim,
  DModel <: Dim,
  Dff <: Dim,
  T <: DType.Floaty]
  (dModel: DModel, dff: Dff, nHeads: NHeads)
  (using Default[T], Default[D], DModel |/ NHeads, RandomSource) {

  type Tn[S <: Shape] = Tensor[S, T, D]
  type MaskTn[M <: Shape] = Tensor[M, Bool, D]
  type Batch[B <: Dim, SeqLen <: Dim] = Tn[(B, SeqLen, DModel)]

  class InputEmbeddings[VocabSize <: Dim](vocabSize: VocabSize) extends Module {
    val embedding = addModule("embedding", Embedding(vocabSize, dModel))

    def apply[S <: Shape, T <: Int32](in: Tensor[S, T, D]): Tn[Append[S, DModel]] = embedding(in) * Math.sqrt(dModel.size.toDouble)
  }

  class PositionalEncoding[SeqLen <: Dim](seqLen: SeqLen, dropoutProb: Double) extends Module {
    val dropout = addModule("dropout", Dropout(dropoutProb))

    val position = Tensor.arangeOf(seqLen).unsqueezeAfter(Last)
    val indices = Tensor.arangeOf(dModel) /|/ 2
    val phase_offset = (Tensor.arangeOf(dModel) % 2) * (Math.PI * 0.5)
    val div_term = exp(indices * (-Math.log(10000.0) / dModel.size))
    val positionalEncodingDeltas = addBuffer("pe", sin(position * div_term + phase_offset))

    def apply[B <: Dim, L <: Dim](in: Batch[B, L]): Batch[B, L] = {
      val deltas = positionalEncodingDeltas(Index.Take(in.sizeOf[L]), Index.All)
      dropout(in + deltas)
    }
  }

  class LayerNormalization(eps: Double = 1e-6) extends Module {
    val alpha = addParameter("alpha", Tensor.ones(1L))
    val bias = addParameter("bias", Tensor.zeros(1L))

    def apply[B <: Dim, SeqLen <: Dim](in: Batch[B, SeqLen]): Batch[B, SeqLen] = {
      val mean = in.meanBy.keepDim(Last) // FIXME verify this, video says "everything after batch" but picks last.
      val std = in.stdBy.keepDim(Last)

      alpha * (in - mean) / (std + eps) + bias
    }
  }

  class FeedForward(dropoutProb: Double) extends Module {
    val l1 = addModule("l1", Linear(dModel, dff))
    val dropout = addModule("dropout", Dropout(dropoutProb))
    val l2 = addModule("l2", Linear(dff, dModel))

    def apply[B <: Shape, SeqLen <: Dim, S <: Shape](in: Tn[S])(using b:Batched1[B, DModel, S]): Tn[B :* DModel] = {
      import b.given
      in ~> l1.apply ~> relu ~> dropout.apply ~> l2.apply
    }
  }

  type AttentionScores[B <: Dim, QSeqLen <: Dim, KVSeqLen <: Dim] = (B, NHeads, QSeqLen, KVSeqLen)

  class MultiHeadAttention(dropoutProb: Double) extends Module {
    val queryWeights = addModule("queryWeights", Linear(dModel, dModel)) // FIXME Maybe there should be no bias here if it's just a mul.
    val keyWeights = addModule("keyWeights", Linear(dModel, dModel))
    val valueWeights = addModule("valueWeights", Linear(dModel, dModel))
    val outputWeights = addModule("outputWeights", Linear(dModel, dModel))
    val dropout = addModule("dropout", Dropout(dropoutProb))

    /** Splits the dModel dimension into NHeads heads, and swap the SeqLen
      * and NHeads dimensions, so each head looks at a sequence of
      * vectors with that head's part of the original DModel. */
    private def splitHeads[B <: Dim, SeqLen <: Dim](b: Batch[B, SeqLen]): Tn[(B, NHeads, SeqLen, DModel / NHeads)] =
      b.view.split[DModel].into(nHeads).transpose[SeqLen, NHeads]

    private def joinHeads[B <: Dim, SeqLen <: Dim](h: Tn[(B, NHeads, SeqLen, DModel / NHeads)]) = {
      h.transpose[NHeads, SeqLen].contiguous.view.merge[DModel / NHeads]
    }

    /** Applies the MultiHeadAttention without a mask (attenting to all input values) */
    def apply[B <: Dim, QSeqLen <: Dim, KVSeqLen <: Dim](
      query: Batch[B, QSeqLen], key: Batch[B, KVSeqLen], value: Batch[B, KVSeqLen]
    ): Batch[B, QSeqLen] =
      apply(query, key, value, None.asInstanceOf[Option[MaskTn[(QSeqLen, KVSeqLen)]]])

    /** @param mask If given, should be true for values we want to give attention to, and false for values to ignore. */
    def apply[B <: Dim, QSeqLen <: Dim, KVSeqLen <: Dim, M <: Shape](
      query: Batch[B, QSeqLen], key: Batch[B, KVSeqLen], value: Batch[B, KVSeqLen], mask: Option[MaskTn[M]]
    ) (using
      Broadcastable[AttentionScores[B, QSeqLen, KVSeqLen], M]
    ): Batch[B, QSeqLen] = {
      val q = query ~> queryWeights.apply ~> splitHeads
      val k = key ~> keyWeights.apply ~> splitHeads
      val v = value ~> valueWeights.apply ~> splitHeads

      val attentionScores = (q `@` k.t / Math.sqrt(dModel.size.toDouble / nHeads.size))
        .when(mask.map(_ #== false))(_.maskedFilled(_, 1e-9))
        .softmax(Last)
        ~> dropout.apply

      // TODO save the attention scores somehow, they're apparently needed for visualization later.
      attentionScores
        `@` v
        ~> joinHeads[B, QSeqLen].apply
        ~> outputWeights.apply
    }
  }

  /** Applies an input through LayerNormalization, then into a
    * sub-layer, and then through a dropout, adding the result to the
    * original input. */
  class ResidualConnection(dropoutProb: Double) extends Module {
    val dropout = addModule("dropout", Dropout(dropoutProb))
    val norm = addModule("norm", new LayerNormalization)

    def apply[B <: Dim, SeqLen <: Dim](sublayer: Batch[B, SeqLen] => Batch[B, SeqLen]): Batch[B, SeqLen] => Batch[B, SeqLen] =
      in => (in ~> norm.apply ~> sublayer ~> dropout.apply) + in
  }

  class EncoderBlock(attention: MultiHeadAttention, feedForward: FeedForward, dropoutProb: Double) extends Module {
    addModule("attention", attention)
    addModule("feedForward", feedForward)
    val residual = addModules("residual", Seq.fill(2)(new ResidualConnection(dropoutProb)))

    def apply[B <: Dim, SeqLen <: Dim, M <: Shape](mask: MaskTn[M])(in: Batch[B, SeqLen])(using
      Broadcastable[AttentionScores[B, SeqLen, SeqLen], M]
    ): Batch[B, SeqLen] = {
      in
        ~> residual(0)(x => attention(x, x, x, Some(mask)))
        ~> residual(1)(feedForward.apply)
    }
  }

  class Encoder(blocks: Seq[EncoderBlock]) extends Module {
    addModules("blocks", blocks)
    val norm = addModule("norm", new LayerNormalization)

    def apply[B <: Dim, SeqLen <: Dim, M <: Shape](mask: MaskTn[M])(in: Batch[B, SeqLen])(using
      Broadcastable[AttentionScores[B, SeqLen, SeqLen], M]
    ): Batch[B, SeqLen] = {
      blocks.foldLeft(in)(_ ~> _(mask)) ~> norm.apply
    }
  }

  class DecoderBlock(
    selfAttention: MultiHeadAttention,
    crossAttention: MultiHeadAttention,
    feedForward: FeedForward,
    dropoutProb: Double) extends Module
  {
    addModule("selfAttention", selfAttention)
    addModule("crossAttention", crossAttention)
    addModule("feedForward", feedForward)
    val residual = addModules("residual", Seq.fill(3)(new ResidualConnection(dropoutProb)))

    def apply[B <: Dim, SrcLen <: Dim, TgtLen <: Dim, EncoderMask <: Shape, DecoderMask <: Shape] (
      encoderOutput: Batch[B, SrcLen], encoderMask: MaskTn[EncoderMask], decoderMask: MaskTn[DecoderMask])(in: Batch[B, TgtLen]
    )(using
      Broadcastable[AttentionScores[B, TgtLen, TgtLen], DecoderMask],
      Broadcastable[AttentionScores[B, TgtLen, SrcLen], EncoderMask]
    ): Batch[B, TgtLen] = {
      in
        ~> residual(0)(x => selfAttention(x, x, x, Some(decoderMask)))
        ~> residual(1)(x => crossAttention(x, encoderOutput, encoderOutput, Some(encoderMask)))
        ~> residual(2)(feedForward.apply)
    }
  }

  class Decoder(blocks: Seq[DecoderBlock]) extends Module {
    addModules("blocks", blocks)
    val norm = addModule("norm", new LayerNormalization)

    def apply[B <: Dim, SrcLen <: Dim, TgtLen <: Dim, EncoderMask <: Shape, DecoderMask <: Shape](
      encoderOutput: Batch[B, SrcLen], encoderMask: MaskTn[EncoderMask], decoderMask: MaskTn[DecoderMask])(in: Batch[B, TgtLen]
    )(using
      Broadcastable[AttentionScores[B, TgtLen, TgtLen], DecoderMask],
      Broadcastable[AttentionScores[B, TgtLen, SrcLen], EncoderMask]
    ): Batch[B, TgtLen] = {
      blocks.foldLeft(in)(_ ~> _(encoderOutput, encoderMask, decoderMask)) ~> norm.apply
    }
  }

  class Projection[VocabSize <: Dim](vocabSize: VocabSize) extends Module {
    val proj = addModule("proj", Linear(dModel, vocabSize))

    def apply[B <: Shape, S <: Shape](in: Tn[S])(using b:Batched1[B, DModel, S]): Tn[B :* VocabSize] = {
      import b.given
      proj(in).log_softmax[VocabSize]
    }
  }

  class Main[
    SrcSeqLen <: Dim, // TODO consider renaming to MaxSrcLen
    TgtSeqLen <: Dim,
    SrcVocabSize <: Dim,
    TgtVocabSize <: Dim
  ](
    encoder: Encoder,
    decoder: Decoder,
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

    parameters.flatMap(_.untyped2D).foreach(init.xavier_uniform)

    def encode[B <: Dim, M <: Shape, T <: Int32, SrcLen <: Dim](src: Tensor[(B, SrcLen), T, D], srcMask: MaskTn[M])(
      using Broadcastable[AttentionScores[B, SrcLen, SrcLen], M]
    ): Batch[B, SrcLen] = {
      src ~> sourceEmb.apply ~> sourcePos.apply ~> encoder(srcMask)
    }

    def decode[B <: Dim, SrcLen <: Dim, TgtLen <: Dim, EM <: Shape, DM <: Shape, T <: Int32]
      (encoderOutput: Batch[B, SrcLen], encoderMask: MaskTn[EM], decoderMask: MaskTn[DM])(tgt: Tensor[(B, TgtLen), T, D])
      (using Broadcastable[AttentionScores[B, TgtLen, TgtLen], DM], Broadcastable[AttentionScores[B, TgtLen, SrcLen], EM])
        : Batch[B, TgtLen] = {
      tgt ~> targetEmb.apply ~> targetPos.apply ~> decoder(encoderOutput, encoderMask, decoderMask)
    }

    def project[B <: Shape, S <: Shape](x: Tn[S])(using b:Batched1[B, DModel, S]): Tn[B :* TgtVocabSize] = projection(x)
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
    DModel <: Dim,
    NHeads <: Dim,
    DFF <: Dim
  ](
    srcVocabSize: SrcVocabSize,
    tgtVocabSize: TgtVocabSize,
    srcSeqLen: SrcSeqLen,
    tgtSeqLen: TgtSeqLen,
    /** Number of dimensions for each embedding in the model, defaults to 512 */
    dModel: DModel,
    /** Size of the hidden feed-forward layer, default to 2048 */
    dFF: DFF,
    /** Number of attention heads (H), default to 8 */
    nHeads: NHeads,
    /** Number of encoder and decoder layers (N), default to 6 */
    coderLayers: Int,
    dropoutProb: Double = 0.1
  )(using rnd:RandomSource, dType:Default[T], device:Default[D])(using DModel |/ NHeads) = {
    val t = new Transformer[D, NHeads, DModel, DFF, T](dModel, dFF, nHeads)
    val srcEmbed = new t.InputEmbeddings(srcVocabSize)
    val tgtEmbed = new t.InputEmbeddings(tgtVocabSize)

    val srcPos = new t.PositionalEncoding(srcSeqLen, dropoutProb)
    val tgtPos = new t.PositionalEncoding(tgtSeqLen, dropoutProb)

    val encoder = new t.Encoder(
      0.until(coderLayers).map { i =>
        new t.EncoderBlock(
          new t.MultiHeadAttention(dropoutProb),
          new t.FeedForward(dropoutProb),
          dropoutProb
        )
      }
    )

    val decoder = new t.Decoder(
      0.until(coderLayers).map { i =>
        new t.DecoderBlock(
          new t.MultiHeadAttention(dropoutProb),
          new t.MultiHeadAttention(dropoutProb),
          new t.FeedForward(dropoutProb),
          dropoutProb
        )
      }
    )

    val projection = new t.Projection(tgtVocabSize)
    t.Main(encoder, decoder, srcEmbed, tgtEmbed, srcPos, tgtPos, projection)
  }
}
