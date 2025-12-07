package net.ypmania.s3torch.internal

class Flatten() {

}

object Flatten {
  // FlattenAfter[D] -> remove all dimensions after D
  // FlattenAll -> remove all dimensions
  // TODO two more operators that flatten UNTIL a dimension.

  type All[S <: Tuple] <: Tuple = S match {
    case EmptyTuple => EmptyTuple
    case Tuple1[dim] => Tuple1[dim]
    case (dimA, dimB) => Tuple1[AddDim[dimA, dimB]]
    case head *: tail => Tuple1[AddDim[head, All[tail]]]
  }

  type First[S <: Tuple] <: Tuple = S match {
    case EmptyTuple => EmptyTuple
    case Tuple1[dim] => Tuple1[dim]
    case (dimA, dimB) => Tuple1[AddDim[dimA, dimB]]
    case dimA *: dimB *: tail => AddDim[dimA, dimB] *: tail
  }
}
