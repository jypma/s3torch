package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Tensor.*
import net.ypmania.s3torch.*

import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch

class ZerosApply[T <: DType](dtype: T, mkTensor: (Array[Long], pytorch.TensorOptions) => pytorch.Tensor) {
  def apply[D1 <: Dim](d1: D1): Tensor[Tuple1[D1], T] = {
    zeros(Seq(d1.size))
  }

  def apply[D1 <: Dim, D2 <: Dim](d1: D1, d2: D2): Tensor[(D1, D2), T] = {
    zeros(Seq(d1.size, d2.size))
  }

  def apply[D1 <: Dim, D2 <: Dim, D3 <: Dim](d1: D1, d2: D2, d3: D3): Tensor[(D1, D2, D3), T] = {
    zeros(Seq(d1.size, d2.size, d3.size))
  }

  // TODO create recursive apply with Tuple as argument and given that will give sizes

  private def zeros[S <: Tuple](size: Seq[Long]): Tensor[S,T] = new Tensor(
    mkTensor(
      size.toArray,
      Torch.tensorOptions(dtype)
    )
  )
}
