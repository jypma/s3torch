package net.ypmania.s3torch.internal

import net.ypmania.s3torch.DType
import net.ypmania.s3torch.DTypeOps
import org.bytedeco.pytorch

import scala.reflect.ClassTag

trait GetNativeValue[S <: Tuple, T <: DType, O] {
  def apply(native: pytorch.Tensor): O
}
object GetNativeValue {
  given d0[T <: DType, O](using ops: DTypeOps[T, O]): GetNativeValue[EmptyTuple, T, O] with {
    def apply(native: pytorch.Tensor) = ops.toScalar(native)
  }
  given d1[T <: DType, O, D1](using ops: DTypeOps[T, O])(using ClassTag[O]): GetNativeValue[Tuple1[D1], T, Seq[O]] with {
    def apply(native: pytorch.Tensor) = ops.toSeq(native)
  }
  given d2[T <: DType, O, D1, D2](using ops: DTypeOps[T, O])(using ClassTag[O]): GetNativeValue[(D1, D2), T, Seq[Seq[O]]] with {
    def apply(native: pytorch.Tensor) = ops.toSeq2D(native)
  }
  given d3[T <: DType, O, D1, D2, D3](using ops: DTypeOps[T, O])(using ClassTag[O]): GetNativeValue[(D1, D2, D3), T, Seq[Seq[Seq[O]]]] with {
    def apply(native: pytorch.Tensor) = ops.toSeq3D(native)
  }
}
