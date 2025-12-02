package net.ypmania.s3torch.internal

import net.ypmania.s3torch.*
import net.ypmania.s3torch.Tensor.*

import org.bytedeco.pytorch.global.torch

trait FromNative[V] {
  type OutputShape <: Tuple
  def apply[T <: DType](value: V, t: T): Tensor[OutputShape, T]
}

object FromNative {
  given FromNative[Double] with {
    type OutputShape = Scalar
    override def apply[T <: DType](value: Double, t: T): Tensor[Scalar, T] = {
      val tensor = torch.scalar_tensor(
        NativeConverters.toScalar(value),
        NativeConverters.tensorOptions(t, Layout.Sparse, Device.CPU, false)
      )
      new Tensor(tensor)
    }
  }
}
