package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Dim._
import net.ypmania.s3torch._

import scala.compiletime.ops.int.-

type Unsplit[S <: Shape, Idx <: Int] <: Shape = (S, Idx) match {
  case (EmptyTuple, 0) => EmptyTuple
  case (next *: DividedDim[originalDim, divisor] *: tail, 1) =>
    next match {
      case divisor => originalDim *: tail
    }
  case (a *: b *: tail, 1) => (a * b) *: tail
  case (head *: tail, idx) => head *: Unsplit[tail, idx - 1]
}
