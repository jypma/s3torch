package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Tensor.*
import net.ypmania.s3torch.*

class ZerosApply[T <: DType] {
  def of[S](using shape: StaticShape[S]): Tensor[shape.OutputShape, T] = {
    val size = shape.size
    ???
  }

  def apply[D1 <: Dim](d1: D1): Tensor[Tuple1[D1], T] = {
    val size = Seq(d1.size)
    println("size is " + size)
    ???
  }

  def apply[D1 <: Dim, D2 <: Dim](d1: D1, d2: D2): Tensor[(D1, D2), T] = {
    val size = Seq(d1.size, d2.size)
    ???
  }

}
