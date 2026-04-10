package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Batched1
import net.ypmania.s3torch.Dim

import Tuple.:*

trait Multinomial[S <: Tuple] {
  type Out[NumSamples <: Dim] <: Tuple
}

object Multinomial {
  given b[S <: Tuple, D <: Dim, B <: Tuple](using b:Batched1[B, D, S]): Multinomial[S] with {
    type Out[NumSamples <: Dim] = B :* NumSamples
  }
}
