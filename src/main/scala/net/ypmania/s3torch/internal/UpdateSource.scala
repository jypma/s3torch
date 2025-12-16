package net.ypmania.s3torch.internal

import org.bytedeco.pytorch

trait UpdateSource[V] {
  def apply(tensor: pytorch.Tensor, idx: pytorch.TensorIndexArrayRef, value: V): Unit
}

object UpdateSource {
  given [V](using toScalar: FromScala.ToScalar[V]): UpdateSource[V] with {
    def apply(tensor: pytorch.Tensor, idx: pytorch.TensorIndexArrayRef, value: V) = {
      tensor.index_put_(idx, toScalar(value))
    }
  }
}
