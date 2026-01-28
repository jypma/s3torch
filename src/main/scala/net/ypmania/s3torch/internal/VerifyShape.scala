package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Shape
import Tuple.Size
import scala.compiletime.ops.int.>=

/** A trait that can be pulled in as given, to check that any match types defining that shape are fully resolved at declaration time. */
trait VerifyShape[S <: Shape]

object VerifyShape {
  given [S <: Shape](using Size[S] >= 0 =:= true): VerifyShape[S] with {}
}
