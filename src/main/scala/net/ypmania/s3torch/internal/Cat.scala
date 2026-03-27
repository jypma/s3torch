package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Shape
import scala.compiletime.ops.int.*

trait Cat[S <: Shape, U <: Shape, Idx <: Int] {}

object Cat {
  given uneq0[D1, D2, T1 <: Shape, T2 <: Shape]: Cat[D1 *: T1, D2 *: T2, 0] with {}
  given uneq[D1, D2, T1 <: Shape, T2 <: Shape, Idx <: Int](using Cat[T1, T2, Idx - 1]): Cat[D1 *: T1, D2 *: T2, Idx] with {}
}
