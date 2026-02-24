package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Dim


trait Untyped[S <: Tuple] {
  type Out <: Tuple
}

object Untyped {
  type ToDim[S <: Tuple] <: Tuple = S match {
    case EmptyTuple => EmptyTuple
    case head *: tail => Dim *: ToDim[tail]
  }
  given [S <: Tuple]: Untyped[S] with {
    type Out = ToDim[S]
  }
}
