package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Tensor.*
import net.ypmania.s3torch.*

import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch
import net.ypmania.s3torch.Dim.DimArg

class ZerosApply[T <: DType](dtype: T, mkTensor: (Array[Long], pytorch.TensorOptions) => pytorch.Tensor) {
  def apply[D1 <: Dim](d1: D1)(using a1: DimArg[D1]): Tensor[Tuple1[a1.Out], T] = {
    zeros(Seq(d1.size))
  }

  // This can't be a tuple and done recursively, as a given's abstract type implementation can not depend on other givens.
  def apply[D1 <: Dim, D2 <: Dim](d1: D1, d2: D2)(using a1: DimArg[D1], a2: DimArg[D2]): Tensor[(a1.Out, a2.Out), T] = {
    zeros(Seq(d1.size, d2.size))
  }

  def apply[D1 <: Dim, D2 <: Dim, D3 <: Dim](d1: D1, d2: D2, d3: D3)(using a1: DimArg[D1], a2: DimArg[D2], a3: DimArg[D3]): Tensor[(a1.Out, a2.Out, a3.Out), T] = {
    zeros(Seq(d1.size, d2.size, d3.size))
  }

  private def zeros[S <: Tuple](size: Seq[Long]): Tensor[S,T] = new Tensor(
    mkTensor(
      size.toArray,
      Torch.tensorOptions(dtype)
    )
  )
}
