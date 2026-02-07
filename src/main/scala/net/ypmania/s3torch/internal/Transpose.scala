package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Shape.AOf
import net.ypmania.s3torch.Shape.BOf
import net.ypmania.s3torch.Shape.BatchOf

import Tuple._

trait Transpose[S <: Tuple, R <: Tuple]

object Transpose {
  given [S <: Tuple, R <: Tuple](using VerifyShape[BatchOf[S]]): Transpose[S, BatchOf[S] ++ (BOf[S], AOf[S])] with {}
}
