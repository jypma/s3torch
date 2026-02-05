package net.ypmania.s3torch.internal

import org.bytedeco.pytorch
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Shape
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.DType.Promoted
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Device

/** Type class that exists for V where V can be the operand to an operation on a tensor with either a scalar (resulting in the same shaped tensor), or another tensor (resulting in Broadcast being applied), with the result being the same or promoted DType of the two tensors. */
trait TensorOperand[S <: Tuple, T <: DType, D <: Device, V] {
  type Out
  def apply(t: Tensor[S, T, D], v: V, withScalar: (pytorch.Tensor, pytorch.Scalar) => pytorch.Tensor, withTensor: (pytorch.Tensor, pytorch.Tensor) => pytorch.Tensor): Out
}

object TensorOperand {
  given scalar[S <: Tuple, T <: DType, D <: Device, V](using toScalar: FromScala.ToScalar[V]): TensorOperand[S, T, D, V] with {
    type Out = Tensor[S, T, D]
    override def apply(t: Tensor[S, T, D], v: V, withScalar: (pytorch.Tensor, pytorch.Scalar) => pytorch.Tensor, withTensor: (pytorch.Tensor, pytorch.Tensor) => pytorch.Tensor) = new Tensor(withScalar(t.native, toScalar(v)))
  }

  given tensor[S <: Tuple, T <: DType, S2 <: Tuple, T2 <: DType, D <: Device, R <: Tuple](using Broadcast[S, S2, R]): TensorOperand[S, T, D, Tensor[S2, T2, D]] with {
    type Out = Tensor[R, Promoted[T, T2], D]
    override def apply(t: Tensor[S, T, D], v: Tensor[S2, T2, D], withScalar: (pytorch.Tensor, pytorch.Scalar) => pytorch.Tensor, withTensor: (pytorch.Tensor, pytorch.Tensor) => pytorch.Tensor) = new Tensor(withTensor(t.native, v.native))
  }
}
