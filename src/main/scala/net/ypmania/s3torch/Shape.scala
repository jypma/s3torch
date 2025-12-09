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
}
