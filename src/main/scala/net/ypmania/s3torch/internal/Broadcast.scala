package net.ypmania.s3torch.internal

import  scala.compiletime.ops.long.*
import net.ypmania.s3torch.*
import Shape.Widen
import net.ypmania.s3torch.Dim.Static
import net.ypmania.s3torch.Dim.One
import net.ypmania.s3torch.Dim.Dynamic
import scala.util.NotGiven

/** Given that shows that S1 and S2 are broadcastable, with shape R as result. */
trait Broadcast[S1 <: Tuple, S2 <: Tuple, R <: Tuple]

object Broadcast {
  /** Given that gives the maximum of both A and B as M */
  trait Max[A <: Dim, B <: Dim, M <: Dim] {
    type Res = M
  }
  trait MaxPrio0 {
    // Fallback when both are not statically known
    given fallback[A <: Dim, B <: Dim](using NotGiven[A <:< Static[?]], NotGiven[B <:< Static[?]]): Max[A, B, Dynamic] with {}
  }
  trait MaxPrio1 extends MaxPrio0 {
    // If both static but the same value, pick either one.
    given eq[AL <: Long, BL <: Long, A <: Static[AL], B <: Static[BL]](using AL =:= BL): Max[A, B, A] with {}
  }
  trait MaxPrio2 extends MaxPrio1 {
    // Either dim is one => pick the other
    given oneA[D <: Dim]: Max[One, D, D] with {}
    given oneB[D <: Dim]: Max[D, One, D] with {}
  }
  object Max extends MaxPrio2 {
    // Same type => pick any
    given same[A <: Dim, B <: Dim](using A =:= B): Max[A, B, A] with {}
  }

  trait MaxEachDim[S1 <: Tuple, S2 <: Tuple, R <: Tuple]

  object MaxEachDim {
    given empty: MaxEachDim[EmptyTuple, EmptyTuple, EmptyTuple] with {}

    given one[A <: Dim, AT <: Tuple, B <: Dim, BT <: Tuple, R <: Dim, RT <: Tuple](using
      MaxEachDim[AT, BT, RT],
      Max[A, B, R]
    ): MaxEachDim[A *: AT, B *: BT, R *: RT] with {}
  }

  given [S1 <: Tuple, S2 <: Tuple, R <: Tuple](using
    MaxEachDim[Widen[S1, S2], Widen[S2, S1], R]
  ): Broadcast[S1, S2, R] with {}

  /** Given that shows that the two shapes are broadcastable */
  trait Apply[S1 <: Tuple, S2 <: Tuple] {
    type Out <: Tuple
  }
  object Apply {
    given[S1 <: Tuple, S2 <: Tuple, R <: Tuple](using Broadcast[S1, S2, R]): Apply[S1, S2] with { type Out = R }
  }
}
