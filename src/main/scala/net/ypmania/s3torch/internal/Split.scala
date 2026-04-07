package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Dim./
import net.ypmania.s3torch.Dim.|/

/** Calculates the result of "splitting", or dividing, the given two dimensions */
trait Split[D, N <: Dim] {
  type Out <: Tuple
}

object Split {
  // TODO add valid splits for nested ProductDim
  given right[A <: Dim, B <: Dim]: Split[Dim.ProductDim[A, B], A] with {
    type Out = (A, B)
  }
  given left[A <: Dim, B <: Dim]: Split[Dim.ProductDim[A, B], B] with {
    type Out = (B, A)
  }
  given divisible[D <: Dim, DV <: Dim](using dv: D |/ DV): Split[D, DV] with {
    type Out = (DV, D / DV)
  }
}
