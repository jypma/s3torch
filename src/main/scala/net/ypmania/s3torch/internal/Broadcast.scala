package net.ypmania.s3torch.internal

import  scala.compiletime.ops.long.*
import net.ypmania.s3torch.*
import Shape.Widen

trait Broadcast[S1 <: Tuple, S2 <: Tuple, R <: Tuple]

object Broadcast {
  trait MaxEachDim[S1 <: Tuple, S2 <: Tuple, R <: Tuple]

  object MaxEachDim {
    given empty: MaxEachDim[EmptyTuple, EmptyTuple, EmptyTuple] with {}

    given one[A <: Dim, AT <: Tuple, B <: Dim, BT <: Tuple, R <: Dim, RT <: Tuple](using
      MaxEachDim[AT, BT, RT],
      Dim.Max[A, B, R]
    ): MaxEachDim[A *: AT, B *: BT, R *: RT] with {}
  }

  given [S1 <: Tuple, S2 <: Tuple, R <: Tuple](using
    MaxEachDim[Widen[S1, S2], Widen[S2, S1], R]
  ): Broadcast[S1, S2, R] with {}

  trait Apply[S1 <: Tuple, S2 <: Tuple] {
    type Out <: Tuple
  }
  object Apply {
    given[S1 <: Tuple, S2 <: Tuple, R <: Tuple](using Broadcast[S1, S2, R]): Apply[S1, S2] with { type Out = R }
  }
}
