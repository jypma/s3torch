package net.ypmania.s3torch.internal

import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch

import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Tensor

trait UpdateSource[V] {
  def apply(tensor: pytorch.Tensor, idx: pytorch.TensorIndexArrayRef, value: V): Unit
}

object UpdateSource {
  given [V](using toScalar: FromScala.ToScalar[V]): UpdateSource[V] with {
    def apply(tensor: pytorch.Tensor, idx: pytorch.TensorIndexArrayRef, value: V) = {
      if (idx.size() == 0 && tensor.sizes().size() == 0) {
        tensor.put(torch.scalar_tensor(toScalar(value), tensor.options))
      } else {
        tensor.index_put_(idx, toScalar(value))
      }
    }
  }
  given [S <: Tuple, T <: DType]: UpdateSource[Tensor[S, T]] with {
    def apply(tensor: pytorch.Tensor, idx: pytorch.TensorIndexArrayRef, value: Tensor[S, T]) = {
      tensor.index_put_(idx, value.native)
    }
  }
}
