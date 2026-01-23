package net.ypmania.s3torch.internal

import scala.compiletime.ops.int.-
import Tuple.*
import net.ypmania.s3torch.*
import net.ypmania.s3torch.Dim.*

type Unsplit[S <: Shape, Idx <: Int] <: Shape = (S, Idx) match {
  case (EmptyTuple, 0) => EmptyTuple
  case (DividedDim[originalDim, _, _] *: _ *: tail, 0) => originalDim *: tail
  case (head *: tail, idx) => head *: Unsplit[tail, idx - 1]
}
