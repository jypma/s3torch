package net.ypmania.s3torch

import scala.compiletime.ops.int._
import Tuple._

/** Can be pulled in as a given to get "Idx" as the index of a selected dimension on a shape, by
  * the dimension's type, First or Last, or compile-time specific numeric index Idx. */
trait Select[S <: Tuple, D, Idx <: Int] {
  type I = Idx
}

object Select {
  /** Selects the first dimension (with index 0) */
  val First = Index.First // reuse so we don't have to make a new name here
  given first[S <: Tuple]: Select[S, First.type, 0] with {}

  /** Selects the last dimension (with the highest index) */
  val Last = Index.Last  // reuse so we don't have to make a new name here
  given last[S <: Tuple]: Select[S, Last.type, Tuple.Size[S] - 1] with {}

  /** Selects a dimension by their exact type. */
  given dimFound[Head <: Dim, Tail <: Tuple]: Select[Head *: Tail, Head, 0] with {}
  import scala.util.NotGiven
  given dimNotFound[Head <: Dim, Tail <: Tuple, D <: Dim, Idx <: Int](using Select[Tail, D, Idx], NotGiven[Head =:= D]): Select[Head *: Tail, D, Idx + 1] with {}

  // TODO add implicit conversion like Dim.fromLongStatic so we can do "3" instead of "Idx(3)"
  /** Selects the dimension at the given index, starting from 0 */
  case class Idx[I <: Int & Singleton](i: I)
  given int[S <: Tuple, I <: Int & Singleton]: Select[S, Idx[I], I] with {}

  /** Selects a specific dimension by type, for which no value might be available. */
  trait Pick[D <: Dim]
  /** Selects a specific dimension by type, for which no value might be available. */
  def dim[D <: Dim]: Pick[D] = new Pick {}
  given atDim[S <: Tuple, D <: Dim, Idx <: Int](using Select[S, D, Idx]): Select[S, Pick[D], Idx] with {}

  extension [S <: Tuple, D, Idx <: Int](d: D)(using Select[S, D, Idx]) {
    /** Compares this selected dimension and the given index into SelectAndIndex, which use used in Tensor.apply. */
    def %[I <: Index](i: I) = SelectAndIndex(d, i)
  }
}

case class SelectAndIndex[D, I <: Index](d: D, i: I)

/** Given that can be pulled in to find a Select[S, D, Idx] for a known S and D. */
trait SelectIdx[S <: Tuple, D] {
  type Idx <: Int
  def idx: Int
}
object SelectIdx {
  given g[S <: Tuple, D, I <: Int](using s:Select[S, D, I], i:ValueOf[I]): SelectIdx[S, D] with {
    type Idx = I
    def idx = i.value
  }

  given concat[B <: Tuple, N <: Tuple, D, I <: Int](using n:Select[N, D, I], i:ValueOf[I], b: ValueOf[Size[B]]): SelectIdx[B ++ N, D] with {
    type Idx = Size[B] + I
    def idx = summon[ValueOf[Size[B]]].value + summon[ValueOf[I]].value
  }

  given append[B <: Tuple, N <: Dim, D](using n:Select[Tuple1[N], D, 0], b: ValueOf[Size[B]]): SelectIdx[B :* N, D] with {
    type Idx = Size[B]
    def idx = summon[ValueOf[Size[B]]].value
  }
}
