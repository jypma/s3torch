package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Dim

import net.ypmania.s3torch.Shape.Elem
import net.ypmania.s3torch.Shape.Replace

/** Proof that [S] and [U] can be concatenated along dimension [Idx], with [Out] as resulting shape. */
trait Cat[S <: Tuple, U <: Tuple, Idx <: Int] {
  type Out <: Tuple
}

trait CatPrio0 {
  // Fallback: Output is Dim.Dynamic (but source shapes must match, except for the target dimension)
  given dyn[S <: Tuple, U <: Tuple, Idx <: Int](using Cat.Verify[S, U, Idx]): Cat[S, U, Idx] with {
    type Out = Replace[S, Dim.Dynamic, Idx]
  }
}

trait CatPrio1 {
  given u_is_dyn[S <: Tuple, U <: Tuple, Idx <: Int](using Elem[U, Idx] <:< Dim.Dynamic, Cat.Verify[S, U, Idx]): Cat[S, U, Idx] with {
    type Out = U
  }
}

object Cat extends CatPrio0 {
  // Given that verifies that S and U are the same shape, except for Idx
  type Verify[S <: Tuple, U <: Tuple, Idx <: Int] = Replace[S, Dim.Dynamic, Idx] =:= Replace[U, Dim.Dynamic, Idx]

  given s_is_dyn[S <: Tuple, U <: Tuple, Idx <: Int](using Elem[S, Idx] <:< Dim.Dynamic, Cat.Verify[S, U, Idx]): Cat[S, U, Idx] with {
    type Out = S
  }
}
