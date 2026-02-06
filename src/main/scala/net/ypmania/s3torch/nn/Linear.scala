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

class Linear[In <: Dim, Out <: Dim, T <: DType] private (native: pytorch.LinearImpl) extends AbstractModule(native) {
  type This[T <: DType] = Linear[In, Out, T]

  def apply[S <: Shape, Idx <: Int](in: Tensor[S, T])(using Tuple.Last[S] =:= In): Tensor[Replace[S, Out, LastIdx[S]],T] = new Tensor(native.forward(in.native))
}

object Linear {
  def apply[In <: Dim, Out <: Dim, T <: DType.Floaty](in: In, out: Out)(using rnd: RandomSource, t: Default[T]): Linear[In, Out, T] =
    rnd(new Linear(new pytorch.LinearImpl(in.size, out.size))).toDType
}
