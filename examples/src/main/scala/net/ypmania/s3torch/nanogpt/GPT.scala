package net.ypmania.s3torch.nanogpt

import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Dim./
import net.ypmania.s3torch.Dim.|/
import net.ypmania.s3torch.Dim.|<=
import net.ypmania.s3torch.Dim.Static
import net.ypmania.s3torch.nn.Module
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Select.dim
import net.ypmania.s3torch.Default
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.nn
import net.ypmania.s3torch.Shape
import net.ypmania.s3torch.Batched
import net.ypmania.s3torch.nn.CrossEntropy
import net.ypmania.s3torch.Index.Last
import net.ypmania.s3torch.Index.First
import net.ypmania.s3torch.Index.TakeLast

class GPT[
  D <: Device,
  T <: DType.Floaty,
  BlockSize <: Dim, // sequence length?
  VocabSize <: Dim,
  LayerCount <: Dim, // n_layer, this might not need to be a Dim, just an Int
  HeadCount <: Dim, // n_head
  EmbeddingCount <: Dim // n_embd
](
  blockSize: BlockSize, // baby: 256
  vocabSize: VocabSize, // baby: 50304, or derived from data set
  layerCount: LayerCount, // baby: 6
  headCount: HeadCount, // baby: 6
  embeddingCount: EmbeddingCount, // baby: 384
  dropoutProb: Double, // baby: 0.0
  bias: Boolean // baby: false
)(using Default[D], Default[T], EmbeddingCount |/ HeadCount) extends Module[D, T] {

  class CausalSelfAttention extends Module {
    // Original model.py uses a single c_attn here for key, query, value. Perhaps for performance reasons?
    // We'll use separate tensors here for clarity.
    val query = addModule("query", nn.Linear(embeddingCount, embeddingCount, bias))
    val key = addModule("key", nn.Linear(embeddingCount, embeddingCount, bias))
    val value = addModule("value", nn.Linear(embeddingCount, embeddingCount, bias))
    val c_proj = addModule("c_proj", nn.Linear(embeddingCount, embeddingCount, bias))
    val attnDropout = addModule("attn_dropout", nn.Dropout(dropoutProb))
    val residDropout = addModule("resid_dropout", nn.Dropout(dropoutProb))

    private def splitHeads[BatchSize <: Dim, SeqLen <: Dim](b: Tensor[(BatchSize, SeqLen, EmbeddingCount), T, D]): Tensor[(BatchSize, HeadCount, SeqLen, EmbeddingCount / HeadCount), T, D] =
      b.view.split(embeddingCount).into(headCount).transpose(dim[SeqLen], headCount)

    private def joinHeads[BatchSize <: Dim, SeqLen <: Dim](h: Tensor[(BatchSize, HeadCount, SeqLen, EmbeddingCount / HeadCount), T, D]) =
      h.transpose(headCount, dim[SeqLen]).contiguous.view.merge[EmbeddingCount / HeadCount]

    def apply[BatchSize <: Dim, SeqLen <: Dim](in: Tensor[(BatchSize, SeqLen, EmbeddingCount), T, D]): in.This = {
      val q = in ~> query.apply ~> splitHeads
      val k = in ~> key.apply ~> splitHeads
      val v = in ~> value.apply ~> splitHeads

      // TODO non-flash attention
      val y = nn.functional.scaledDotProductAttention(q, k, v, dropout = if (isTraining) dropoutProb else 0.0)
      y ~> joinHeads[BatchSize, SeqLen].apply ~> c_proj.apply ~> residDropout.apply
    }
  }

  class MLP extends Module {
    val c_fc = addModule("c_fc", nn.Linear(embeddingCount, embeddingCount * 4, bias))
    val c_proj = addModule("c_proj", nn.Linear(embeddingCount * 4, embeddingCount, bias))
    val dropout = addModule("drop", nn.Dropout(dropoutProb))

    def apply[BatchSize <: Dim, SeqLen <: Dim](in: Tensor[(BatchSize, SeqLen, EmbeddingCount), T, D]): in.This = {
      in ~> c_fc.apply ~> nn.functional.gelu ~> c_proj.apply ~> dropout.apply
    }
  }

  class Block extends Module {
    val ln_1 = addModule("ln_1", nn.LayerNorm(Shape(embeddingCount), bias = bias))
    val attn = addModule("attn", new CausalSelfAttention)
    val ln_2 = addModule("ln_2", nn.LayerNorm(Shape(embeddingCount), bias = bias))
    val mlp = addModule("mlp", new MLP)

    def apply[BatchSize <: Dim, SeqLen <: Dim](x: Tensor[(BatchSize, SeqLen, EmbeddingCount), T, D]): x.This = {
      val x1 = x + attn(ln_1(x))
      //x += attn(ln_1(x))

      val x2 = x1 + mlp(ln_2(x1))
      // x += mlp(ln_2(x))

      x2
    }
  }

  // TODO add weight typing once we reimplement nn.Embedding
  val wte = addModule("wte", nn.Embedding(vocabSize, embeddingCount))
  val wpe = addModule("wpe", nn.Embedding(blockSize, embeddingCount))
  val drop = addModule("drop", nn.Dropout(dropoutProb))
  val h = addModules("h", Seq.fill(layerCount.size.toInt)(new Block))
  val ln_f = addModule("ln_f", nn.LayerNorm(Shape(embeddingCount), bias = bias))
  val lm_head = addModule("lm_head", nn.Linear(embeddingCount, vocabSize))

  initWeights()

  private def initWeights(): Unit = {
    // TODO fancy weight initialization, somehow without being able to instanceof the modules.
  }

  private def embedding[BatchSize <: Dim, Len <: Dim, I <: DType.Int64](idx: Tensor[(BatchSize, Len), I, D])(using Len |<= BlockSize): Tensor[(BatchSize, Len, EmbeddingCount), T, D] = {
    val len = idx.sizeOf(dim[Len])
    val pos = Tensor.arangeOf(len)
    val tok_emb = wte(idx)
    val pos_emb = wpe(pos)

    var x = drop(tok_emb + pos_emb)

    for (block <- h) {
      x = block(x)
    }

    ln_f(x)
  }

  // "forward" with targets
  def train[BatchSize <: Dim, Len <: Dim, I <: DType.Int64](idx: Tensor[(BatchSize, Len), I, D], targets: Tensor[(BatchSize, Len), I, D])(using Len |<= BlockSize) = {
    val expected = targets.view.merge[Len]
    val logits = lm_head(embedding(idx))
    val actual = logits.view.merge[Len]
    val loss = CrossEntropy(actual, expected.to(DType.int64))

    // FIXME we don't use the logits here
    (logits, loss)
  }

  // "forward" without targets
  def infer[BatchSize <: Dim, Len <: Dim, T <: DType.Int64](idx: Tensor[(BatchSize, Len), T, D])(using Len |<= BlockSize, Dim.One |<= Len) = {
    val x = embedding(idx)
    val onlyLast = x(dim[Len] % TakeLast(Dim.One)) // Only infer for the last element of the vector
    val logits = lm_head(onlyLast)

    logits.squeeze(Dim.One)
  }

  // TODO rewrite, this actually extends Length with one, so |<= isn't guaranteed.
  def extend[Len <: Dim.Dynamic, T <: DType.Int64](idx: Tensor[Tuple1[Len], T, D])(using Len |<= BlockSize, Dim.One |<= Len) = {

    val predicted = infer(idx.unsqueezeBefore(First)).squeeze(First)
    val probs = predicted.softmax(vocabSize)
    val next = probs.multinomial(Dim.One).toDTypeOf(idx)

    idx.cat(next)(dim[Len])
  }

}
