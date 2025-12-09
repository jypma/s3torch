package net.ypmania.s3torch.internal

import  scala.compiletime.ops.long.*
import net.ypmania.s3torch.*
import Shape.Widen

type Broadcast[S1 <: Tuple, S2 <: Tuple] = Broadcast.MaxEachDim[Widen[S1, S2], Widen[S2, S1]]

object Broadcast {
  type MaxEachDim[S1 <: Tuple, S2 <: Tuple] <: Tuple = (S1, S2) match {
    case (EmptyTuple, EmptyTuple) => EmptyTuple
    case (d1 *: tail1, d2 *: tail2) => Dim.Max[d1, d2] *: MaxEachDim[tail1, tail2]
  }
}
