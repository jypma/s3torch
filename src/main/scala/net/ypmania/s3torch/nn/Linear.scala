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

class Linear[In <: Dim, Out <: Dim] private (native: pytorch.LinearImpl) extends AbstractModule(native) {
  def apply[S <: Shape, T <: DType, Idx <: Int](in: Tensor[S, T])(using Tuple.Last[S] =:= In): Tensor[Replace[S, Out, LastIdx[S]],T] = new Tensor(native.forward(in.native))
}

object Linear {
  def apply[T <: DType](using dtype: Default[T]) = new Apply(dtype.value)

  class Apply[T <: DType](dtype: T) {
    def apply[In <: Dim, Out <: Dim](in: In, out: Out): Linear[In, Out] = new Linear(new pytorch.LinearImpl(in.size, out.size)).to(dtype)
  }

}
