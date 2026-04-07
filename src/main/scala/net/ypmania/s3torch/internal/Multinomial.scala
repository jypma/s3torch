package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Dim

trait Multinomial[S <: Tuple] {
  type Out[NumSamples <: Dim] <: Tuple
}

object Multinomial {
  /** pytorch: If input is a vector, out is a vector of size num_samples. */
  given [D1 <: Dim]: Multinomial[Tuple1[D1]] with {
    type Out[NumSamples <: Dim] = Tuple1[NumSamples]
  }

  /** pytorch: If input is a matrix with m rows, out is an matrix of shape (m × num_samples). */
  given [D1 <: Dim, D2 <: Dim]: Multinomial[(D1, D2)] with {
    type Out[NumSamples <: Dim] = (D1, NumSamples)
  }

}
