package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Tensor.*
import net.ypmania.s3torch.*

import org.bytedeco.pytorch.global.torch

class ZerosApply[T <: DType](dtype: T) {
  def apply[D1 <: Dim](d1: D1): Tensor[Tuple1[D1], T] = {
    zeros(Seq(d1.size))
  }

  def apply[D1 <: Dim, D2 <: Dim](d1: D1, d2: D2): Tensor[(D1, D2), T] = {
    zeros(Seq(d1.size, d2.size))
  }

  private def zeros[S <: Tuple](size: Seq[Long]): Tensor[S,T] = new Tensor(
    torch.torch_zeros(
      size.toArray,
      Torch.tensorOptions(dtype)
    )
  )
}
