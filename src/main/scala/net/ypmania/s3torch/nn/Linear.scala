package net.ypmania.s3torch.nn

import net.ypmania.s3torch.Batched1
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Default
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.RandomSource
import net.ypmania.s3torch.Tensor
import org.bytedeco.pytorch

import Tuple._

class Linear[In <: Dim, Out <: Dim, D <: Device, T <: DType] private (native: pytorch.LinearImpl) extends AbstractModule[D, T](native) {
  type This[D <: Device, T <: DType] = Linear[In, Out, D, T]

  def apply[S <: Tuple, B <: Tuple, T <: DType, Idx <: Int](in: Tensor[S, T, D])(using Batched1[B, In, S]): in.Shaped[B ++ Tuple1[Out]] =
    new Tensor(native.forward(in.native))
}

object Linear {
  def apply[In <: Dim, Out <: Dim, D <: Device, T <: DType.Floaty](in: In, out: Out)(using rnd: RandomSource, t: Default[T], d: Default[D]): Linear[In, Out, D, T] =
    rnd(new Linear(new pytorch.LinearImpl(in.size, out.size))).toDeviceDType
}
