package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Shape

import scala.compiletime.ops.int.>=

import Tuple._

/** A trait that can be pulled in as given, to check that any match types defining that shape are fully resolved at declaration time. */
trait VerifyShape[S <: Shape]

trait VerifyShapePrio0 {
  given knownStatic[S <: Shape](using Size[S] >= 0 =:= true): VerifyShape[S] with {}
}

trait VerifyShapePrio1 extends VerifyShapePrio0 {
  given knownGiven[S <: Shape](using ValueOf[Size[S]]): VerifyShape[S] with {}
}

object VerifyShape extends VerifyShapePrio1  {
  given concatGiven[S <: Shape, N <: Shape](using ValueOf[Size[S]], ValueOf[Size[N]]): VerifyShape[S ++ N] with {}
  given appendGiven[S <: Shape, N <: Dim](using ValueOf[Size[S]]): VerifyShape[S :* N] with {}
}
