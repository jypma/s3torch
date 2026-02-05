package net.ypmania.s3torch.internal

import org.bytedeco.pytorch
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Shape
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.DType.Promoted
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Device

trait TensorOperandApply[S <: Tuple, T <: DType, D <: Device, V] {
  def apply(t: Tensor[S, T, D], v: V, withScalar: (pytorch.Tensor, pytorch.Scalar) => Unit, withTensor: (pytorch.Tensor, pytorch.Tensor) => Unit): Unit
}

object TensorOperandApply {
  given scalar[S <: Tuple, T <: DType, D <: Device, V](using toScalar: FromScala.ToScalar[V]): TensorOperandApply[S, T, D, V] with {
    override def apply(t: Tensor[S, T, D], v: V, withScalar: (pytorch.Tensor, pytorch.Scalar) => Unit, withTensor: (pytorch.Tensor, pytorch.Tensor) => Unit) = withScalar(t.native, toScalar(v))
  }

  given tensor[S <: Tuple, T <: DType, S2 <: Tuple, T2 <: DType, D <: Device](using Broadcast[S, S2, S]): TensorOperandApply[S, T, D, Tensor[S2, T2, D]] with {
    override def apply(t: Tensor[S, T, D], v: Tensor[S2, T2, D], withScalar: (pytorch.Tensor, pytorch.Scalar) => Unit, withTensor: (pytorch.Tensor, pytorch.Tensor) => Unit) = withTensor(t.native, v.native)
  }
}
