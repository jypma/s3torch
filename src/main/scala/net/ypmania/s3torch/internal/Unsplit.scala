package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Dim._
import net.ypmania.s3torch._

/** Calculates the result of "unsplitting", or multiplying, the given two dimensions */
trait Unsplit[D1, D2] {
  type Out <: Dim
  def apply(d1: D1, d2: D2): Out
}

trait UnsplitLowPrio {
  /** Low priority given that returns the product of the two dimensions */
  inline given prod[A <: Dim, B <: Dim]: Unsplit[A, B] with {
    type Out = (A * B)
    override def apply(a: A, b: B) = ProductDim(a,b)
  }
}

object Unsplit extends UnsplitLowPrio {
  /** High priority given that matches a previously split dimension, undoing the split. */
  inline given divided[Divisor <: Dim, Original <: Dim]: Unsplit[Divisor, DividedDim[Original, Divisor]] with {
    type Out = Original
    override def apply(div: Divisor, dim: DividedDim[Original, Divisor]): Original = dim.original
  }
}
