package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Shape
import net.ypmania.s3torch.Dim

/** Given that is available when shape I can be safely cast to shape O, by them having the same dimensions, and each dimension in O being either Dynamic, or the same as I. */
trait CanCastShape[I <: Shape, O]

object CanCastShape {
  trait CanCastDim[I, O]
  object CanCastDim {
    given same[T <: Dim]: CanCastDim[T, T] with {}
    given toDyn[I <: Dim, O <: Dim.Dynamic]: CanCastDim[I, O] with {}
  }

  given CanCastShape[EmptyTuple, EmptyTuple] with {}
  given [I <: Dim, O <: Dim, IT <: Shape, OT <: Shape](using CanCastDim[I, O], CanCastShape[IT, OT]): CanCastShape[I *: IT, O *: OT] with {}
}

trait CanShaped[I <: Shape, O] {
  type Out <: Shape
}

object CanShaped {
  given tuple[I <: Shape, O <: Shape](using CanCastShape[I, O]): CanShaped[I, O] with { type Out = O }
  given single[I <: Dim, O <: Dim](using CanCastShape.CanCastDim[I, O]): CanShaped[Tuple1[I], O] with { type Out = Tuple1[O] }
}
