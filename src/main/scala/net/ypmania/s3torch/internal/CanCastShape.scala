package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Dim

/** Given that is available when shape I can be safely cast to shape O, by them having the same dimensions, and each dimension in O being either Dynamic, or the same as I. */
trait CanCastShape[I <: Tuple, O]

object CanCastShape {
  trait CanCastDim[I, O]

  trait CanCastDimPrio0 {
    given toDyn[I <: Dim, O <: Dim.Dynamic]: CanCastDim[I, O] with {}
  }
  object CanCastDim extends CanCastDimPrio0 {
    given same[T <: Dim]: CanCastDim[T, T] with {}
  }

  given CanCastShape[EmptyTuple, EmptyTuple] with {}
  given [I <: Dim, O <: Dim, IT <: Tuple, OT <: Tuple](using CanCastDim[I, O], CanCastShape[IT, OT]): CanCastShape[I *: IT, O *: OT] with {}
}

trait CanShaped[I <: Tuple, O] {
  type Out <: Tuple
}

trait CanShapedPrio0 {
  given tuple[I <: Tuple, O <: Tuple](using CanCastShape[I, O]): CanShaped[I, O] with { type Out = O }
}

object CanShaped extends CanShapedPrio0 {
  given single[I <: Dim, O <: Dim](using CanCastShape.CanCastDim[I, O]): CanShaped[Tuple1[I], O] with { type Out = Tuple1[O] }
}
