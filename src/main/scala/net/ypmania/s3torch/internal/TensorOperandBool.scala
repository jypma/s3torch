package net.ypmania.s3torch.internal
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Tensor
import org.bytedeco.pytorch

/** Type class that exists for V where V can be the operand to an operation on a tensor with either a scalar (resulting in the same shaped tensor), or another tensor (resulting in Broadcast being applied), with the result being a DType of Bool. */
trait TensorOperandBool[S <: Tuple, T <: DType, D <: Device, V] {
  type Out
  def apply(t: Tensor[S, T, D], v: V, withScalar: (pytorch.Tensor, pytorch.Scalar) => pytorch.Tensor, withTensor: (pytorch.Tensor, pytorch.Tensor) => pytorch.Tensor): Out
}

object TensorOperandBool {
  given scalar[S <: Tuple, T <: DType, D <: Device, V](using toScalar: FromScala.ToScalar[V]): TensorOperandBool[S, T, D, V] with {
    type Out = Tensor[S, DType.Bool.type, D]
    override def apply(t: Tensor[S, T, D], v: V, withScalar: (pytorch.Tensor, pytorch.Scalar) => pytorch.Tensor, withTensor: (pytorch.Tensor, pytorch.Tensor) => pytorch.Tensor) = new Tensor(withScalar(t.native, toScalar(v)))
  }

  given tensor[S <: Tuple, T <: DType, S2 <: Tuple, T2 <: DType, D <: Device, R <: Tuple](using Broadcast[S, S2, R]): TensorOperandBool[S, T, D, Tensor[S2, T2, D]] with {
    type Out = Tensor[R, DType.Bool.type, D]
    override def apply(t: Tensor[S, T, D], v: Tensor[S2, T2, D], withScalar: (pytorch.Tensor, pytorch.Scalar) => pytorch.Tensor, withTensor: (pytorch.Tensor, pytorch.Tensor) => pytorch.Tensor) = new Tensor(withTensor(t.native, v.native))
  }
}
