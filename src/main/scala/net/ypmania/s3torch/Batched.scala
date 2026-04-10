package net.ypmania.s3torch

import Tuple._
import scala.compiletime.ops.int.+

/** Proof that shape [S] is in fact the concatenation of (potentially call-site unknown) batch dimensions [B] with actual, known dimensions [T] */
abstract class Batched[B <: Tuple, T <: Tuple, S <: Tuple](using b:ValueOf[Size[B]], t:ValueOf[Size[T]]) {
  given batchSize:ValueOf[Size[B]] = summon[ValueOf[Size[B]]]
  given tailSize:ValueOf[Size[T]] = summon[ValueOf[Size[T]]]

  given fromBatch[Dt <: DType, Dv <: Device]: Conversion[Tensor[B ++ T, Dt, Dv], Tensor[S, Dt, Dv]] = _.asInstanceOf[Tensor[S, Dt, Dv]]
  given fromBatch1[Dt <: DType, Dv <: Device, T1](using Tuple1[T1] =:= T): Conversion[Tensor[B :* T1, Dt, Dv], Tensor[S, Dt, Dv]] = _.asInstanceOf[Tensor[S, Dt, Dv]]
}

type Batched1[B <: Tuple, D <: Dim, S <: Tuple] = Batched[B, Tuple1[D], S]

trait BatchedPrio0 {
  given reduceTail[B <: Tuple, T <: Tuple, S <: Tuple, D1](using b: Batched[B, D1 *: T, S]): Batched[B :* D1, T, S](using
    b = ValueOf((b.batchSize.value + 1).asInstanceOf[Size[B :* D1]]),
    t = ValueOf((b.tailSize.value - 1).asInstanceOf[Size[T]])
  ) with {}
}

trait BatchedPrio1 {
  given append2[B <: Tuple, S <: Tuple, D1, D2](using b:Batched[B, Tuple1[D1], S]): Batched[B, (D1, D2), S :* D2](using b = b.batchSize) with {}
  given concat[B <: Tuple, T <: Tuple](using ValueOf[Size[B]], ValueOf[Size[T]]): Batched[B, T, B ++ T] with {}
  given shapeConcat[B <: Tuple, T <: Tuple](using ValueOf[Size[B]], ValueOf[Size[T]]): Batched[B, T, Shape.Concat[B, T]] with {}
}

object Batched extends BatchedPrio0 with BatchedPrio1 {
  given append[B <: Tuple, D](using ValueOf[Size[B]]): Batched[B, Tuple1[D], B :* D] with {}

  given d1[D <: Dim]: Batched[EmptyTuple, Tuple1[D], Tuple1[D]] with {}

  given d20[D1 <: Dim, D2 <: Dim]: Batched[EmptyTuple, (D1, D2), (D1, D2)] with {}

  given d30[D1 <: Dim, D2 <: Dim, D3 <: Dim]: Batched[EmptyTuple, (D1, D2, D3), (D1, D2, D3)] with {}

  given d40[D1 <: Dim, D2 <: Dim, D3 <: Dim, D4 <: Dim]: Batched[EmptyTuple, (D1, D2, D3, D4), (D1, D2, D3, D4)] with {}

}
