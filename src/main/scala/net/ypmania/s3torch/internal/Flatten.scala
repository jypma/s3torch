package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Shape
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Dim._

trait Flatten[S <: Shape] {
  type Out <: Dim
}

trait FlattenLowPrio {
  given fallback[S <: Shape]: Flatten[S] with { type Out = Dim }
}

object Flatten extends FlattenLowPrio {
  // TODO Create a recursive variant of this that actually compiles
  given d1[D1 <: Dim]: Flatten[Tuple1[D1]] with { type Out = D1 }
  given d2[D1 <: Dim, D2 <: Dim]: Flatten[(D1, D2)] with { type Out = D1 * D2 }
  given d3[D1 <: Dim, D2 <: Dim, D3 <: Dim]: Flatten[(D1, D2, D3)] with { type Out = D1 * D2 * D3 }
  given d4[D1 <: Dim, D2 <: Dim, D3 <: Dim, D4 <: Dim]: Flatten[(D1, D2, D3, D4)] with { type Out = D1 * D2 * D3 * D4 }
}
