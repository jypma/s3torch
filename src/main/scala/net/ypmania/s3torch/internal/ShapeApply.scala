package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Shape

/** Contructs shape-like objects from multiple dimension arguments */
trait ShapeApply[T[_ <: Tuple]] {
  def apply[D1 <: Dim](d1: D1): T[Tuple1[D1]] = {
    create(Seq(d1.size))
  }

  def apply[D1 <: Dim, D2 <: Dim](d1: D1, d2: D2): T[(D1, D2)] = {
    create(Seq(d1.size, d2.size))
  }

  def apply[D1 <: Dim, D2 <: Dim, D3 <: Dim](d1: D1, d2: D2, d3: D3): T[(D1, D2, D3)] = {
    create(Seq(d1.size, d2.size, d3.size))
  }

  def apply[S <: Tuple](s: Shape[S]): T[S] = create(s.size)

  def apply[S <: Tuple](s: S)(using sizes: Shape.Sizes[S]): T[S] = create(sizes.value(s))

  protected def create[S <: Tuple](size: Seq[Long]): T[S]
}
