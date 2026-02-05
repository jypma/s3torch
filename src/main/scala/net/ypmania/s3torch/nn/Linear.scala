package net.ypmania.s3torch.nn

import org.bytedeco.pytorch
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Shape
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Default

import Tuple.*
import Shape.*
import scala.compiletime.ops.int.*
import net.ypmania.s3torch.RandomSource
import AbstractModule.CreationDType
import net.ypmania.s3torch.Device

class Linear[In <: Dim, Out <: Dim, D <: Device, T <: DType] private (native: pytorch.LinearImpl) extends AbstractModule[D, T](native) {
  type This[D <: Device, T <: DType] = Linear[In, Out, D, T]

  def apply[S <: Shape, T <: DType, D <: Device, Idx <: Int](in: Tensor[S, T, D])(using Tuple.Last[S] =:= In): in.Shaped[Replace[S, Out, LastIdx[S]]] =
    new Tensor(native.forward(in.native))
}

object Linear {
  def apply[In <: Dim, Out <: Dim, D <: Device, T <: DType.Floaty](in: In, out: Out)(using rnd: RandomSource, t: Default[T], d: Default[D]): Linear[In, Out, D, T] =
    rnd(new Linear(new pytorch.LinearImpl(in.size, out.size))).toDeviceDType
}
