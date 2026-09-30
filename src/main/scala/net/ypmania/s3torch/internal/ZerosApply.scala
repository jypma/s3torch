package net.ypmania.s3torch.internal

import net.ypmania.s3torch._
import org.bytedeco.pytorch

/** Constructs a Tensor from a shape (for methods like Tensor.zeros) */
class ZerosApply[T <: DType, D <: Device](dtype: T, device: D, mkTensor: (Array[Long], pytorch.TensorOptions) => pytorch.Tensor) extends ShapeApply[[S <: Tuple] =>> Tensor[S, T, D]] {
  protected def create[S <: Tuple](size: Seq[Long]): Tensor[S, T, D] = new Tensor(
    mkTensor(
      size.toArray,
      Torch.tensorOptions(dtype, device)
    )
   )
}
