package net.ypmania.s3torch.internal

import net.ypmania.s3torch.*
import net.ypmania.s3torch.Tensor.*

import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch

trait FromNative[V] {
  type OutputShape <: Tuple
  def apply[T <: DType](value: V, t: T): Tensor[OutputShape, T]
}

object FromNative {
  type ScalarLike = Boolean | Byte | Short | Int | Long | Float | Double

  given [N <: ScalarLike]: FromNative[N] with {
    type OutputShape = Scalar
    override def apply[T <: DType](value: N, t: T): Tensor[Scalar, T] = {
      val tensor = torch.scalar_tensor(
        toScalar(value),
        Torch.tensorOptions(t)
      )
      new Tensor(tensor)
    }
  }

  private def toScalar[N <: ScalarLike](x: N): pytorch.Scalar = x match {
    case x: Boolean => pytorch.AbstractTensor.create(x).item()
    case x: Byte    => pytorch.Scalar(x)
    case x: Short   => pytorch.Scalar(x)
    case x: Int     => pytorch.Scalar(x)
    case x: Long    => pytorch.Scalar(x)
    case x: Float   => pytorch.Scalar(x)
    case x: Double  => pytorch.Scalar(x)
  }
}
