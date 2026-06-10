package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Dim.Dynamic
import net.ypmania.s3torch.Dim.One
import net.ypmania.s3torch.Dim.Static
import net.ypmania.s3torch.Dim.|<=
import net.ypmania.s3torch._

import scala.util.NotGiven

import Shape.Widen
import Broadcast._
import Tuple._

/** Given that shows that S1 and S2 are broadcastable, with shape R as result. */
trait Broadcast[S1 <: Tuple, S2 <: Tuple, R <: Tuple]

type Broadcastable[S1 <: Tuple, S2 <: Tuple] = Broadcast[S1, S2, S1]

trait BroadcastPrio0 {
  given [S1 <: Tuple, S2 <: Tuple, R <: Tuple](using
    MaxEachDim[Widen[S1, S2], Widen[S2, S1], R]
  ): Broadcast[S1, S2, R] with {}
}

object Broadcast extends BroadcastPrio0 {
  given b1[B <: Tuple, A1 <: Dim, B1 <: Dim, R <: Tuple](using Broadcast[Tuple1[A1], Tuple1[B1], R]): Broadcast[B :* A1, Tuple1[B1], Shape.Concat[B, R]] with {}
  given bb[B <: Tuple, A1 <: Dim, A2 <: Dim, B1 <: Dim, B2 <: Dim, R <: Tuple](using Broadcast[(A1, A2), (B1, B2), R]): Broadcast[B ++ (A1, A2), (B1, B2), Shape.Concat[B, R]] with {}


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
  trait MaxPrio3 extends MaxPrio2 {
    given lt[A <: Dim, B <: Dim](using A |<= B): Max[A, B, B] with {}
  }
  object Max extends MaxPrio3 {
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
}
