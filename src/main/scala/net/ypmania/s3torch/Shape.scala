package net.ypmania.s3torch

import scala.compiletime.ops.int._

import Tuple._

type Shape = Tuple

object Shape {
  type Scalar = EmptyTuple

  type Elem[X <: Shape, N <: Int] = Tuple.Elem[X, N]

  type Size[X <: Shape] = Tuple.Size[X]

  /** Just some simplifications over plain Tuple.Concat to get the worst types out of the way. */
  type Concat[X <: Shape, Y <: Shape] = Y match {
    case EmptyTuple => X
    case d1 *: EmptyTuple => X :* d1
    case _ => Tuple.Concat[X, Y]
  }

  /** The shape of S is widened to accomodate the dimensions of To, by prepending static dimensions of one. */
  type Widen[S <: Tuple, To <: Tuple] <: Tuple = Size[S] < Size[To] match {
    case true => Widen[Dim.One *: S, To]
    case false => S
  }

  type InsertBefore[S <: Shape, I <: Dim, Idx <: Int] <: Shape = Idx match {
    case -1 => S
    case _ => Tuple.Concat[
      Tuple.Take[S, Idx],
      I *: Tuple.Drop[S, Idx]
    ]
  }
  type InsertAfter[S <: Shape, I <: Dim, Idx <: Int] = InsertBefore[S, I, Idx + 1]

  type Remove[S <: Shape, Idx <: Int] <: Shape = (Idx, S) match {
    case (_, EmptyTuple) => EmptyTuple
    case (0, dim *: tail) => tail
    case (0, _) => S
    case (_, dim *: tail) => dim *: Remove[tail, Idx - 1]
  }

  type RemoveLast[S <: Shape] = Remove[S, Size[S] - 1]

  /** Replaces the dimension at [Idx] with the dimension [I] */
  type Replace[S <: Shape, I, Idx <: Int] <: Shape = Idx match {
    case -1 => S
    case _ => Tuple.Concat[
      Tuple.Take[S, Idx],
      I *: Tuple.Drop[S, Idx + 1]
    ]
  }

  /** Replaces the dimension at [Idx] with all dimensions in tuple [I] */
  type ReplaceWithTuple[S <: Shape, I <: Tuple, Idx <: Int] <: Shape = Idx match {
    case -1 => S
    case _ => Tuple.Concat[
      Tuple.Take[S, Idx],
      I ++ Tuple.Drop[S, Idx + 1]
    ]
  }

  /** Replaces the dimension at [Idx] and [Idx - 1] with all dimensions in tuple [I] */
  type Replace2WithTuple[S <: Shape, I <: Tuple, Idx <: Int] <: Shape = Idx match {
    case -1 => S
    case _ => Tuple.Concat[
      Tuple.Take[S, Idx - 1],
      I ++ Tuple.Drop[S, Idx + 1]
    ]
  }

  /** The index of the last dimension in the shape */
  type LastIdx[S <: Shape] = Tuple.Size[S] - 1

  /** Swaps two dimensions  */
  type Swap[S <: Shape, I1 <: Int, I2 <: Int] = (I1 < I2) match {
    case true => internal.SwapLT[S, I1, I2]
    case false => internal.SwapLT[S, I2, I1]
  }

  object internal {
    type SwapLT[S <: Shape, I1 <: Int, I2 <: Int] = I1 match {
      case -1 => S
      case _ =>
        Take[S, I1] ++ (Elem[S, I2] *: Replace[Drop[S, I1 + 1], Elem[S, I1], I2 - I1 - 1])
    }
  }

  trait Sizes[S <: Shape] {
    def value(s: S): Seq[Long]
  }
  object Sizes {
    given Sizes[EmptyTuple] with { def value(s: EmptyTuple) = Seq.empty }
    given [D <: Dim, Tail <: Shape](using tail: Sizes[Tail]): Sizes[D *: Tail] with { def value(s: D *: Tail) = s.head.size +: tail.value(s.tail) }
  }

  /** Given that is available if S1 has the same number of dimensions as S2 */
  type SameSize[S1 <: Shape, S2 <: Shape] = Size[S1] =:= Size[S2]

  /** The batch dimension(s) of S */
  type BatchOf[S <: Shape] = Take[S, Size[S] - 2]
  /** The "A" matrix dimension, i.e. the first one */
  type AOf[S <: Shape] = Last[Init[S]]
  /** The "B" matrix dimension, i.e. the second one */
  type BOf[S <: Shape] = Last[S]

  /** Given that can be pulled in to get the last dimension of [S], as a Dim. */
  trait LastDim[S <: Shape] {
    type D <: Dim
  }
  object LastDim {
    given [A <: Dim]: LastDim[Tuple1[A]] with {
      type D = A
    }
    given [S <: Tuple, A <: Dim](using l: LastDim[S]): LastDim[A *: S] with {
      type D = l.D
    }
  }
}
