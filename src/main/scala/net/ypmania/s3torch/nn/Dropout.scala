package net.ypmania.s3torch.nn

import org.bytedeco.pytorch

import net.ypmania.s3torch.*

import Tuple.:*

class Dropout[D <: Device, T <: DType] private (native: pytorch.DropoutImpl)(using RandomSource) extends AbstractModule[D, T](native) {
  type This[D <: Device, T <: DType] = Dropout[D, T]

  private val rnd = summon[RandomSource].fork // Fork, so subsequent invocations of apply() have different results, but still reproducable.

  def apply[S <: Shape, T <: DType, D <: Device](in: Tensor[S, T, D]): in.This = rnd(new Tensor(native.forward(in.native)))
}

object Dropout {
  /** Creates a new dropout layer, with the given probability */
  def apply[D <: Device, T <: DType.Floaty](probability: Double = 0.5)(using rnd: RandomSource, t: Default[T], d: Default[D]): Dropout[D, T] =
    new Dropout(new pytorch.DropoutImpl(probability)).toDeviceDType
}
