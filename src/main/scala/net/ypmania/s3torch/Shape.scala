package net.ypmania.s3torch

import scala.compiletime.ops.int._

import Tuple._
import net.ypmania.s3torch.internal.ShapeApply

case class Shape[S <: Tuple] private[s3torch] (size: Seq[Long]) {
  /** Returns the size of one dimension selected by D, as a Dim.Ref (since we can't create an actual instance of a Dim). */
  def sizeOf[D](d: D)(using sel: SelectIdx[S, D]): Dim.Ref[Elem[S, sel.Idx]] = Dim.Ref(size(sel.idx))

  /** Returns the size of one dimension typed D, using [dim] to create the instance of D holding the result. */
  def sizeOf[D <: Dim, Idx <: Int](dim: Long => D)(using sel: SelectIdx[S, D]): D = dim(size(sel.idx))
}

object Shape extends ShapeApply[[S <: Tuple] =>> Shape[S]] {
  protected def create[S <: Tuple](size: Seq[Long]): Shape[S] = new Shape[S](size)

  type Scalar = EmptyTuple

  type Elem[X <: Tuple, N <: Int] = Tuple.Elem[X, N]

  type Size[X <: Tuple] = Tuple.Size[X]

  /** Just some simplifications over plain Tuple.Concat to get the worst types out of the way. */
  type Concat[X <: Tuple, Y <: Tuple] = Y match {
    case EmptyTuple => X
    case d1 *: EmptyTuple => X :* d1
    case _ => Tuple.Concat[X, Y]
  }

  /** The shape of S is widened to accomodate the dimensions of To, by prepending static dimensions of one. */
  type Widen[S <: Tuple, To <: Tuple] <: Tuple = Size[S] < Size[To] match {
    case true => Widen[Dim.One *: S, To]
    case false => S
  }

  type InsertBefore[S <: Tuple, I <: Dim, Idx <: Int] <: Tuple = Idx match {
    case -1 => S
    case _ => Tuple.Concat[
      Tuple.Take[S, Idx],
      I *: Tuple.Drop[S, Idx]
    ]
  }
  type InsertAfter[S <: Tuple, I <: Dim, Idx <: Int] = InsertBefore[S, I, Idx + 1]

  type Remove[S <: Tuple, Idx <: Int] <: Tuple = (Idx, S) match {
    case (_, EmptyTuple) => EmptyTuple
    case (0, dim *: tail) => tail
    case (0, _) => S
    case (_, dim *: tail) => dim *: Remove[tail, Idx - 1]
  }

  type RemoveLast[S <: Tuple] = Remove[S, Size[S] - 1]

  /** Replaces the dimension at [Idx] with the dimension [I] */
  type Replace[S <: Tuple, I, Idx <: Int] <: Tuple = Idx match {
    case -1 => S
    case _ => Tuple.Concat[
      Tuple.Take[S, Idx],
      I *: Tuple.Drop[S, Idx + 1]
    ]
  }

  /** Replaces the dimension at [Idx] with all dimensions in tuple [I] */
  type ReplaceWithTuple[S <: Tuple, I <: Tuple, Idx <: Int] <: Tuple = Idx match {
    case -1 => S
    case _ => Tuple.Concat[
      Tuple.Take[S, Idx],
      I ++ Tuple.Drop[S, Idx + 1]
    ]
  }

  /** Replaces the dimension at [Idx] and [Idx - 1] with all dimensions in tuple [I] */
  type Replace2WithTuple[S <: Tuple, I <: Tuple, Idx <: Int] <: Tuple = Idx match {
    case -1 => S
    case _ => Tuple.Concat[
      Tuple.Take[S, Idx - 1],
      I ++ Tuple.Drop[S, Idx + 1]
    ]
  }

  /** The index of the last dimension in the shape */
  type LastIdx[S <: Tuple] = Tuple.Size[S] - 1

  /** Swaps two dimensions  */
  type Swap[S <: Tuple, I1 <: Int, I2 <: Int] = (I1 < I2) match {
    case true => internal.SwapLT[S, I1, I2]
    case false => internal.SwapLT[S, I2, I1]
  }

  object internal {
    type SwapLT[S <: Tuple, I1 <: Int, I2 <: Int] = I1 match {
      case -1 => S
      case _ =>
        Take[S, I1] ++ (Elem[S, I2] *: Replace[Drop[S, I1 + 1], Elem[S, I1], I2 - I1 - 1])
    }
  }

  // TODO: Rename to Shape.Magnet, and allow Shape itself here, and perhaps Tensor as well (not just Tuple).
  trait Sizes[S <: Tuple] {
    def value(s: S): Seq[Long]
  }
  object Sizes {
    given Sizes[EmptyTuple] with { def value(s: EmptyTuple) = Seq.empty }
    given [D <: Dim, Tail <: Tuple](using tail: Sizes[Tail]): Sizes[D *: Tail] with { def value(s: D *: Tail) = s.head.size +: tail.value(s.tail) }
  }

  /** Given that is available if S1 has the same number of dimensions as S2 */
  type SameSize[S1 <: Tuple, S2 <: Tuple] = Size[S1] =:= Size[S2]

  /** The batch dimension(s) of S */
  type BatchOf[S <: Tuple] = Take[S, Size[S] - 2]
  /** The "A" matrix dimension, i.e. the first one */
  type AOf[S <: Tuple] = Last[Init[S]]
  /** The "B" matrix dimension, i.e. the second one */
  type BOf[S <: Tuple] = Last[S]

  /** Given that can be pulled in to get the last dimension of [S], as a Dim. */
  trait LastDim[S <: Tuple] {
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
