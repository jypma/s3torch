package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Tensor.*
import net.ypmania.s3torch.*

import org.bytedeco.pytorch.global.torch
import net.ypmania.s3torch.Dim.DimArg

class ZerosApply[T <: DType](dtype: T) {
  def apply[D1 <: Dim](d1: D1)(using a1: DimArg[D1]): Tensor[Tuple1[a1.Out], T] = {
    zeros(Seq(d1.size))
  }

  // TODO Find a nice recursive way (args to tuple?) to express higher dimensions
  def apply[D1 <: Dim, D2 <: Dim](d1: D1, d2: D2)(using a1: DimArg[D1], a2: DimArg[D2]): Tensor[(a1.Out, a2.Out), T] = {
    zeros(Seq(d1.size, d2.size))
  }

  private def zeros[S <: Tuple](size: Seq[Long]): Tensor[S,T] = new Tensor(
    torch.torch_zeros(
      size.toArray,
      Torch.tensorOptions(dtype)
    )
  )
}
