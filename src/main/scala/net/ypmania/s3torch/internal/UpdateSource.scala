package net.ypmania.s3torch.internal

import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Tensor
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch

trait UpdateSource[V, D <: Device] {
  def apply(tensor: pytorch.Tensor, idx: pytorch.TensorIndexArrayRef, value: V): Unit
}

object UpdateSource {
  given [V, D <: Device](using toScalar: FromScala.ToScalar[V]): UpdateSource[V, D] with {
    def apply(tensor: pytorch.Tensor, idx: pytorch.TensorIndexArrayRef, value: V) = {
      if (idx.size() == 0 && tensor.sizes().size() == 0) {
        tensor.put(torch.scalar_tensor(toScalar(value), tensor.options))
      } else {
        tensor.index_put_(idx, toScalar(value))
      }
    }
  }
  // The UpdateSource source tensor must be on the same device as we're updating.
  given [S <: Tuple, T <: DType, D <: Device]: UpdateSource[Tensor[S, T, D], D] with {
    def apply(tensor: pytorch.Tensor, idx: pytorch.TensorIndexArrayRef, value: Tensor[S, T, D]) = {
      tensor.index_put_(idx, value.native)
    }
  }
}
