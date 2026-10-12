package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Dim

trait Multinomial[S <: Tuple] {
  type Out[NumSamples <: Dim] <: Tuple
}

object Multinomial {
  // Torch will give a "prob_dist must be 1 or 2 dim" error, which is undocumented: dimension must be 1 or 2

  given d2[D1 <: Dim, D2 <: Dim]: Multinomial[(D1, D2)] with {
    type Out[NumSamples <: Dim] = (D1, NumSamples)
  }
  given d1[D1 <: Dim]: Multinomial[Tuple1[D1]] with {
    type Out[NumSamples <: Dim] = Tuple1[NumSamples]
  }
}
