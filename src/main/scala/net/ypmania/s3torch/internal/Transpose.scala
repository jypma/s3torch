package net.ypmania.s3torch.internal

import Tuple.*
import scala.compiletime.ops.int.-
import net.ypmania.s3torch.Shape.BatchOf
import net.ypmania.s3torch.Shape.AOf
import net.ypmania.s3torch.Shape.BOf

trait Transpose[S <: Tuple, R <: Tuple]

object Transpose {
  given [S <: Tuple, R <: Tuple](using VerifyShape[BatchOf[S]]): Transpose[S, BatchOf[S] ++ (BOf[S], AOf[S])] with {}
}
