package net.ypmania.s3torch

import Tuple.*

abstract class Batched[B <: Tuple, T <: Tuple, S <: Tuple](using ValueOf[Size[B]], ValueOf[Size[T]]) {
  given batchSize:ValueOf[Size[B]] = summon[ValueOf[Size[B]]]
}

type Batched1[B <: Tuple, D <: Dim, S <: Tuple] = Batched[B, Tuple1[D], S]

object Batched {
  given concat[B <: Tuple, T <: Tuple](using ValueOf[Size[B]], ValueOf[Size[T]]): Batched[B, T, B ++ T] with {}

  given d1[D <: Dim]: Batched[EmptyTuple, Tuple1[D], Tuple1[D]] with {}

  given d20[D1 <: Dim, D2 <: Dim]: Batched[EmptyTuple, (D1, D2), (D1, D2)] with {}
  given d21[D1 <: Dim, D2 <: Dim]: Batched[Tuple1[D1], Tuple1[D2], (D1, D2)] with {}

  given d31[D1 <: Dim, D2 <: Dim, D3 <: Dim]: Batched[Tuple1[D1], (D2, D3), (D1, D2, D3)] with {}
  given d32[D1 <: Dim, D2 <: Dim, D3 <: Dim]: Batched[(D1, D2), Tuple1[D3], (D1, D2, D3)] with {}
}
