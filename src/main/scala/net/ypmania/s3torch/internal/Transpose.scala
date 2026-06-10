package net.ypmania.s3torch.internal

import Tuple._
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Batched

trait Transpose[S <: Tuple, R <: Tuple]

object Transpose {
  given [S <: Tuple, B <: Tuple, Row <: Dim, Col <: Dim](using b: Batched[B, (Row, Col), S]): Transpose[S, B :* Col :* Row] with {}
}
