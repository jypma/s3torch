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

  type InsertAfter[S <: Shape, I <: Dim, D <: Dim] <: Shape = IndexOf[S, D] match { // Take, Drop, Concat
    case -1 => S
    case _ => Tuple.Concat[
      Tuple.Take[S, IndexOf[S, D]],
      I *: Tuple.Drop[S, IndexOf[S, D]]
    ]
  }
}
