package net.ypmania.s3torch

import scala.compiletime.ops.int.*
import Tuple.*

type Shape = Tuple

object Shape {
  type Scalar = EmptyTuple

  /** The shape of S is widened to accomodate the dimensions of To, by prepending static dimensions of one. */

  type Widen[S <: Tuple, To <: Tuple] <: Tuple = Size[S] < Size[To] match {
    case true => Widen[Dim.One *: S, To]
    case false => S
  }

  type IndexOf[S <: Shape, D <: Dim] <: Int = S match {
    case D *: tail => 0
    case EmptyTuple => -1
    case _ *: tail => 1 + IndexOf[tail, D]
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

  type Replace[S <: Shape, I <: Dim, Idx <: Int] <: Shape = Idx match {
    case -1 => S
    case _ => Tuple.Concat[
      Tuple.Take[S, Idx],
      I *: Tuple.Drop[S, Idx + 1]
    ]
  }

  type LastIdx[S <: Shape] = Tuple.Size[S] - 1

  /** Can be pulled in as a given to get "Idx" as the index of a selected dimension on a shape, by
    * the dimension's type, First or Last, or compile-time specific numeric index Idx. */
  trait Select[S <: Shape, D, Idx <: Int]

  object Select {
    case object First
    given [S <: Shape]: Select[S, First.type, 0] with {}

    case object Last
    given [S <: Shape]: Select[S, Last.type, Tuple.Size[S] - 1] with {}

    given [S <: Shape, D <: Dim]: Select[S, D, IndexOf[S, D]] with {}

    case class Idx[I <: Int & Singleton](i: I)
    given int[S <: Shape, I <: Int & Singleton]: Select[S, Idx[I], I] with {}
  }
}
