package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Shape
import Tuple.*
import net.ypmania.s3torch.Shape.Widen

trait MatMul[S1 <: Tuple, S2 <: Tuple, R <: Tuple]

trait MatMulPrio0 {
  // Multiply two batches that have the same dimensionality (not necessarily the same dimensions)
  // FIXME this is not correct, it messes up dimensions 1 and 2.
  /*
  given sameDimBatch[A <: Dim, AT <: Tuple, B <: Dim, BT <: Tuple, R <: Dim, RT <: Tuple](using
    MatMul[AT, BT, RT],
    Dim.Max[A, B, R]
   ): MatMul[A *: AT, B *: BT, R *: RT] with {}
   */
}

// Comments come from https://docs.pytorch.org/docs/stable/generated/torch.matmul.html
object MatMul extends MatMulPrio0 {
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

  // - Multiply batch with 1D vector
  // Scala doesn't seem to search for givens if the same Tuple type occurs in >1 position.
  given d1m1[A1 <: Dim, A <: Dim, B <: Dim]: MatMul[(A1, A,  B), Tuple1[B], (A1, A)] with {}
  given d1m2[A1 <: Dim, A2 <: Dim, A <: Dim, B <: Dim]: MatMul[(A1, A2, A,  B), Tuple1[B], (A1, A2, A)] with {}

  // Missing:
  // - Multiply 1D vector with batch
  // - Multiply batch with 2D matrix
  // - Multiply 2D matrix with batch
  // - Multiple batches of different dimensions
}
