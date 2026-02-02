package net.ypmania.s3torch.internal

import org.bytedeco.pytorch
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Shape
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.DType.Promoted
import net.ypmania.s3torch.Dim

trait TensorOperandApply[S <: Tuple, T <: DType, V] {
  def apply(t: Tensor[S, T], v: V, withScalar: (pytorch.Tensor, pytorch.Scalar) => Unit, withTensor: (pytorch.Tensor, pytorch.Tensor) => Unit): Unit
}

object TensorOperandApply {
  given scalar[S <: Tuple, T <: DType, V](using toScalar: FromScala.ToScalar[V]): TensorOperandApply[S, T, V] with {
    override def apply(t: Tensor[S, T], v: V, withScalar: (pytorch.Tensor, pytorch.Scalar) => Unit, withTensor: (pytorch.Tensor, pytorch.Tensor) => Unit) = withScalar(t.native, toScalar(v))
  }

  given tensor[S <: Tuple, T <: DType, S2 <: Tuple, T2 <: DType](using Broadcast[S, S2, S]): TensorOperandApply[S, T, Tensor[S2, T2]] with {
    override def apply(t: Tensor[S, T], v: Tensor[S2, T2], withScalar: (pytorch.Tensor, pytorch.Scalar) => Unit, withTensor: (pytorch.Tensor, pytorch.Tensor) => Unit) = withTensor(t.native, v.native)
  }
}
