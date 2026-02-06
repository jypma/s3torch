package net.ypmania.s3torch.nn

import org.bytedeco.pytorch

import net.ypmania.s3torch.*

import Tuple.:*

class Dropout private (native: pytorch.DropoutImpl)(using RandomSource) extends AbstractModule(native) {
  type This[T <: DType] = Dropout

  private val rnd = summon[RandomSource].fork // Fork, so subsequent invocations of apply() have different results, but still reproducable.

  def apply[S <: Shape, T <: DType](in: Tensor[S,T]): Tensor[S,T] = rnd(new Tensor(native.forward(in.native)))
}

object Dropout {
  /** Creates a new dropout layer, with the given probability */
  def apply(probability: Double = 0.5)(using rnd: RandomSource) = new Dropout(new pytorch.DropoutImpl(probability))
}
