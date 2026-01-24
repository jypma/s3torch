package net.ypmania.s3torch.internal

import org.bytedeco.pytorch
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Shape
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.DType.Promoted
import net.ypmania.s3torch.Dim

/** Type class that exists for V where V can be the operand to an operation on a tensor with either a scalar (resulting in the same shaped tensor), or another tensor (resulting in Broadcast being applied) */
trait TensorOperand[S <: Tuple, T <: DType, V] {
  type Out
  def apply(t: Tensor[S, T], v: V, withScalar: (pytorch.Tensor, pytorch.Scalar) => pytorch.Tensor, withTensor: (pytorch.Tensor, pytorch.Tensor) => pytorch.Tensor): Out
}

object TensorOperand {
  given scalar[S <: Tuple, T <: DType, V](using toScalar: FromScala.ToScalar[V]): TensorOperand[S, T, V] with {
    type Out = Tensor[S, T]
    override def apply(t: Tensor[S, T], v: V, withScalar: (pytorch.Tensor, pytorch.Scalar) => pytorch.Tensor, withTensor: (pytorch.Tensor, pytorch.Tensor) => pytorch.Tensor) = new Tensor(withScalar(t.native, toScalar(v)))
  }

  given tensor[S <: Tuple, T <: DType, S2 <: Tuple, T2 <: DType, R <: Tuple](using Broadcast[S, S2, R]): TensorOperand[S, T, Tensor[S2, T2]] with {
    type Out = Tensor[R, Promoted[T, T2]]
    override def apply(t: Tensor[S, T], v: Tensor[S2, T2], withScalar: (pytorch.Tensor, pytorch.Scalar) => pytorch.Tensor, withTensor: (pytorch.Tensor, pytorch.Tensor) => pytorch.Tensor) = new Tensor(withTensor(t.native, v.native))
  }
}
