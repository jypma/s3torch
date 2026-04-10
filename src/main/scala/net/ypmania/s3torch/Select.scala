package net.ypmania.s3torch

import scala.compiletime.ops.int._
import Tuple._

/** Can be pulled in as a given to get "Idx" as the index of a selected dimension on a shape, by
  * the dimension's type, First or Last, or compile-time specific numeric index Idx. */
trait Select[S <: Shape, D, Idx <: Int] {
  type I = Idx
}

object Select {
  /** Selects the first dimension (with index 0) */
  case object Start
  given first[S <: Shape]: Select[S, Start.type, 0] with {}

  /** Selects the last dimension (with the highest index) */
  case object End
  given last[S <: Shape]: Select[S, End.type, Tuple.Size[S] - 1] with {}

  /** Selects a dimension by their exact type. */
  given dimFound[Head <: Dim, Tail <: Shape]: Select[Head *: Tail, Head, 0] with {}
  import scala.util.NotGiven
  given dimNotFound[Head <: Dim, Tail <: Shape, D <: Dim, Idx <: Int](using Select[Tail, D, Idx], NotGiven[Head =:= D]): Select[Head *: Tail, D, Idx + 1] with {}

  // TODO add implicit conversion like Dim.fromLongStatic so we can do "3" instead of "Idx(3)"
  /** Selects the dimension at the given index, starting from 0 */
  case class Idx[I <: Int & Singleton](i: I)
  given int[S <: Shape, I <: Int & Singleton]: Select[S, Idx[I], I] with {}

  /** Selects a specific dimension by type, for which no value might be available. */
  trait Pick[D <: Dim]
  /** Selects a specific dimension by type, for which no value might be available. */
  def dim[D <: Dim]: Pick[D] = new Pick {}
  given atDim[S <: Shape, D <: Dim, Idx <: Int](using Select[S, D, Idx]): Select[S, Pick[D], Idx] with {}

  extension [S <: Shape, D, Idx <: Int](d: D)(using Select[S, D, Idx]) {
    /** Compares this selected dimension and the given index into SelectAndIndex, which use used in Tensor.apply. */
    def %[I <: Index](i: I) = SelectAndIndex(d, i)
  }
}

case class SelectAndIndex[D, I <: Index](d: D, i: I)

/** Given that can be pulled in to find a Select[S, D, Idx] for a known S and D. */
trait SelectIdx[S <: Shape, D] {
  type Idx <: Int
  def idx: Int
}
object SelectIdx {
  given g[S <: Shape, D, I <: Int](using s:Select[S, D, I], i:ValueOf[I]): SelectIdx[S, D] with {
    type Idx = I
    def idx = i.value
  }

  given concat[B <: Shape, N <: Shape, D, I <: Int](using n:Select[N, D, I], i:ValueOf[I], b: ValueOf[Size[B]]): SelectIdx[B ++ N, D] with {
    type Idx = Size[B] + I
    def idx = summon[ValueOf[Size[B]]].value + summon[ValueOf[I]].value
  }

  given append[B <: Shape, N <: Dim, D](using n:Select[Tuple1[N], D, 0], b: ValueOf[Size[B]]): SelectIdx[B :* N, D] with {
    type Idx = Size[B]
    def idx = summon[ValueOf[Size[B]]].value
  }
}
