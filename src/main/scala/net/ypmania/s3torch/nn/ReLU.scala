package net.ypmania.s3torch.nn

import net.ypmania.s3torch._
import org.bytedeco.pytorch

class ReLU[D <: Device, T <: DType] private (native: pytorch.ReLUImpl) extends AbstractModule[D, T](native) {
  type This[D <: Device, T <: DType] = ReLU[D, T]

  def apply[S <: Shape, T <: DType, D <: Device](in: Tensor[S, T, D]): in.This = new Tensor(native.forward(in.native))
}

object ReLU {
  /** Creates a new ReLU layer, which applies the rectified linear unit function element-wise. */
  def apply[D <: Device, T <: DType.Floaty](using t: Default[T], d: Default[D]): ReLU[D, T] =
    new ReLU(new pytorch.ReLUImpl()).toDeviceDType
}
