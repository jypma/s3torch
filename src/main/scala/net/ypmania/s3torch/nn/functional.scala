package net.ypmania.s3torch.nn

import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Batched

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.TensorOptional
import org.bytedeco.pytorch.DoubleOptional

import Tuple.++
import org.bytedeco.pytorch.GELUOptions

object functional {
  /** Creates a dot product attention with a causal mask */
  def scaledDotProductAttention[B <: Tuple, L <: Dim, S <: Dim, E <: Dim, T <: DType, D <: Device, QS <: Tuple, KS <: Tuple, VS <: Tuple](
    q: Tensor[QS, T, D],
    k: Tensor[KS, T, D],
    v: Tensor[VS, T, D],
    dropout: Double = 0.0
  )(using
    qb: Batched[B, (L, E), QS],
    kb: Batched[B, (S, E), KS],
    vb: Batched[B, (S, E), VS]
  ): Tensor[B ++ (L, E), T, D] = {
    new Tensor(
      torch.scaled_dot_product_attention(q.native, k.native, v.native, new TensorOptional(), dropout, /*isCausal*/true, new DoubleOptional(), /*enableGQA*/false)
    )
  }

  sealed trait GeluApproximate(val native: String)
  object GeluApproximate {
    case object none extends GeluApproximate("none")
    case object tanh extends GeluApproximate("tanh")
  }

  /** Applies the Gaussian Error Linear Units function */
  def gelu[S <: Tuple, T <: DType, D <: Device](in: Tensor[S, T, D]): in.This = {
    gelu(in, GeluApproximate.none)
  }

  /** Applies the Gaussian Error Linear Units function */
  def gelu[S <: Tuple, T <: DType, D <: Device](in: Tensor[S, T, D], approx: GeluApproximate): in.This = {
    val opts = new GELUOptions()
    opts.approximate(approx.native)
    new Tensor(torch.gelu(in.native, opts))
  }
}
