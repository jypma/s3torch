package net.ypmania.s3torch.nn

import net.ypmania.s3torch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.TensorOptional
import net.ypmania.s3torch.internal.ShapeApply

class LayerNorm[D <: Device, T <: DType, W <: Tuple](shape: Shape[W], useBias: Boolean, eps: Double)(using Default[D], Default[T]) extends Module[D, T] {
  override type This[D <: Device, T <: DType] = LayerNorm[D, T, W]

  val weight = addParameter("weight", Tensor.ones(shape))
  val bias = Option.when(useBias)(addParameter("bias", Tensor.zeros(shape)))

  /** Incoming shape S must be batch dimensions B plus weight dimensions W */
  def apply[B <: Tuple, S <: Tuple, T <: DType, D <: Device](in: Tensor[S, T, D])(using b: Batched[B, W, S]): in.This = {
    val nw = new TensorOptional(weight.native)
    val nb = bias.map(b => new TensorOptional(b.native)).getOrElse(new TensorOptional())

    new Tensor(torch.layer_norm(in.native, shape.size.toArray, nw, nb, eps, true))
  }
}

// This is implemented as a custom module, rather than using libtorch's LayerNormImpl, since the latter doesn't expose customizing
// whether a bias should be present.
object LayerNorm {
  /** Creates a new LayerNorm layer, e.g. LayerNorm(Shape(2L, 3L)) */
  def apply[D <: Device, T <: DType.Floaty, W <: Tuple](shape: Shape[W], bias: Boolean = true, eps: Double = 1e-5)(using t: Default[T], d: Default[D]): LayerNorm[D,T,W] =
    new LayerNorm(shape, bias, eps)

  /** Creates a new LayerNorm layer with no bias and eps of 1e-5, e.g. LayerNorm(2L, 3L) */
  def apply[D <: Device, T <: DType.Floaty](using t: Default[T], d: Default[D]) = new ShapeApply[[W <: Tuple] =>> LayerNorm[D, T, W]] {
    protected def create[W <: Tuple](size: Seq[Long]): LayerNorm[D, T, W] = new LayerNorm(new Shape[W](size), true, 1e-5)
  }
}
