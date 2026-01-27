package net.ypmania.s3torch.internal

import scala.compiletime.ops.int.-
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Shape
import Tuple.*
import net.ypmania.s3torch.Shape.Widen

trait MatMul[S1 <: Tuple, S2 <: Tuple, R <: Tuple]

// Comments come from https://docs.pytorch.org/docs/stable/generated/torch.matmul.html
object MatMul {
  //If both tensors are 1-dimensional, the dot product (scalar) is returned.
  given d1[D <: Dim]: MatMul[Tuple1[D], Tuple1[D], Shape.Scalar] with {}

  //If both arguments are 2-dimensional, the matrix-matrix product is returned (of matrix AxZ matmul ZxB, giving AxB).
  given d2[A <: Dim, Z <: Dim, B <: Dim]: MatMul[(A, Z), (Z, B), (A, B)] with {}

  // If the first argument is 1-dimensional and the second argument is
  // 2-dimensional, a 1 is prepended to its dimension for the purpose
  // of the matrix multiply. After the matrix multiply, the prepended
  // dimension is removed.
  // So, 1xA matmul AxB giving 1xB, but only returning B
  given d1a[A <: Dim, B <: Dim]: MatMul[Tuple1[A], (A, B), Tuple1[B]] with {}

  // If the first argument is 2-dimensional and the second argument is
  // 1-dimensional, the matrix-vector product is returned.
  // So, AxB matmul B, returning A
  given d1b[A <: Dim, B <: Dim]: MatMul[(A, B), Tuple1[B], Tuple1[A]] with {}

  /** The batch dimension(s) of S */
  type BatchOf[S <: Tuple] = Take[S, Size[S] - 2]
  /** The "A" matrix dimension, i.e. the first one */
  type AOf[S <: Tuple] = Last[Init[S]]
  /** The "B" matrix dimension, i.e. the second one */
  type BOf[S <: Tuple] = Last[S]

  // Broadcast for matrices
  given bt[D1 <: Tuple, D2 <: Tuple, R <: Tuple, BR <: Tuple](using
    MatMul[(AOf[D1], BOf[D1]), (AOf[D2], BOf[D2]), R],
    Broadcast[BatchOf[D1], BatchOf[D2], BR]
  ): MatMul[D1, D2, BR ++ R] with {}

  // Multiply batch with  vector
  given mulWith1D[S <: Tuple, D <: Dim, R <: Tuple](using
    MatMul[(AOf[S], BOf[S]), Tuple1[D], R]
  ): MatMul[S, Tuple1[D], BatchOf[S] ++ R] with {}

  // Multiply vector with batch
  given mul1DWith[S <: Tuple, D <: Dim, R <: Tuple](using
    MatMul[Tuple1[D], (AOf[S], BOf[S]), R]
  ): MatMul[Tuple1[D], S, BatchOf[S] ++ R] with {}

  // Multiply batch with 2D matrix
  given mulWith2D[S <: Tuple, A <: Dim, B <: Dim, R <: Tuple](using
    MatMul[(AOf[S], BOf[S]), (A, B), R]
  ): MatMul[S, (A, B), BatchOf[S] ++ R] with {}

  // Multiply 2D matrix with batch
  given mul2DWith[S <: Tuple, A <: Dim, B <: Dim, R <: Tuple](using
    MatMul[(A, B), (AOf[S], BOf[S]), R]
  ): MatMul[(A, B), S, BatchOf[S] ++ R] with {}
}
