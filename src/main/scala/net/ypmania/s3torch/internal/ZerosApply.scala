package net.ypmania.s3torch.internal

import net.ypmania.s3torch._
import org.bytedeco.pytorch

class ZerosApply[T <: DType, D <: Device](dtype: T, device: D, mkTensor: (Array[Long], pytorch.TensorOptions) => pytorch.Tensor) {
  def apply[D1 <: Dim](d1: D1): Tensor[Tuple1[D1], T, D] = {
    zeros(Seq(d1.size))
  }

  def apply[D1 <: Dim, D2 <: Dim](d1: D1, d2: D2): Tensor[(D1, D2), T, D] = {
    zeros(Seq(d1.size, d2.size))
  }

  def apply[D1 <: Dim, D2 <: Dim, D3 <: Dim](d1: D1, d2: D2, d3: D3): Tensor[(D1, D2, D3), T, D] = {
    zeros(Seq(d1.size, d2.size, d3.size))
  }

  def apply[S <: Shape](s: S)(using sizes: Shape.Sizes[S]): Tensor[S, T, D] = zeros(sizes.value(s))

  private def zeros[S <: Tuple](size: Seq[Long]): Tensor[S, T, D] = new Tensor(
    mkTensor(
      size.toArray,
      Torch.tensorOptions(dtype, device)
    )
  )
}
