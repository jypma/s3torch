package net.ypmania.s3torch

import scala.compiletime.ops.int.*
import Tuple.*
import net.ypmania.s3torch.Dim.IndexOfDivided

type Shape = Tuple

object Shape {
  type Scalar = EmptyTuple

  type Elem[X <: Shape, N <: Int] = Tuple.Elem[X, N]

  /** The shape of S is widened to accomodate the dimensions of To, by prepending static dimensions of one. */
  type Widen[S <: Tuple, To <: Tuple] <: Tuple = Size[S] < Size[To] match {
    case true => Widen[Dim.One *: S, To]
    case false => S
  }

  type IndexOf[S <: Shape, D <: Dim] <: Int = S match {
    case D *: tail => 0
    case _ *: tail => 1 + IndexOf[tail, D]
  }

  type InsertBefore[S <: Shape, I <: Dim, Idx <: Int] <: Shape = Idx match {
    case -1 => S
    case _ => Tuple.Concat[
      Tuple.Take[S, Idx],
      I *: Tuple.Drop[S, Idx]
    ]
  }
  type InsertAfter[S <: Shape, I <: Dim, Idx <: Int] = InsertBefore[S, I, Idx + 1]

  type Remove[S <: Shape, Idx <: Int] <: Shape = (Idx, S) match {
    case (_, EmptyTuple) => EmptyTuple
    case (0, dim *: tail) => tail
    case (0, _) => S
    case (_, dim *: tail) => dim *: Remove[tail, Idx - 1]
  }

  /** Replaces the dimension at [Idx] with the dimension [I] */
  type Replace[S <: Shape, I <: Dim, Idx <: Int] <: Shape = Idx match {
    case -1 => S
    case _ => Tuple.Concat[
      Tuple.Take[S, Idx],
      I *: Tuple.Drop[S, Idx + 1]
    ]
  }

  /** Replaces the dimension at [Idx] with all dimensions in tuple [I] */
  type ReplaceWithTuple[S <: Shape, I <: Tuple, Idx <: Int] <: Shape = Idx match {
    case -1 => S
    case _ => Tuple.Concat[
      Tuple.Take[S, Idx],
      I ++ Tuple.Drop[S, Idx + 1]
    ]
  }
  type LastIdx[S <: Shape] = Tuple.Size[S] - 1

  /** Swaps two dimensions  */
  type Swap[S <: Shape, I1 <: Int, I2 <: Int] = (I1 < I2) match {
    case true => internal.SwapLT[S, I1, I2]
    case false => internal.SwapLT[S, I2, I1]
  }

  object internal {
    type SwapLT[S <: Shape, I1 <: Int, I2 <: Int] = I1 match {
      case -1 => S
      case _ =>
        Take[S, I1] ++ (Elem[S, I2] *: Replace[Drop[S, I1 + 1], Elem[S, I1], I2 - I1 - 1])
    }
  }

  trait Is2D[S <: Shape] {
    type D1 <: Dim
    type D2 <: Dim
  }
  given [A <: Dim, B <: Dim]: Is2D[(A, B)] with {
    type D1 = A
    type D2 = B
  }

  // TODO ----------- move Select trait to its own file --------------

  /** Can be pulled in as a given to get "Idx" as the index of a selected dimension on a shape, by
    * the dimension's type, First or Last, or compile-time specific numeric index Idx. */
  trait Select[S <: Shape, D, Idx <: Int] {
    type I = Idx
  }

  object Select {
    /** Selects the first dimension (with index 0) */
    case object First
    given first[S <: Shape]: Select[S, First.type, 0] with {}

    /** Selects the last dimension (with the highest index) */
    case object Last
    given last[S <: Shape]: Select[S, Last.type, Tuple.Size[S] - 1] with {}

    // This will only select a dim where D is an explicit type. If D
    // is a type parameter, it's only known to be a subclass of Dim,
    // so you can't use match types with that.
    /** Selects a dimension by their exact type. */
    given dim[S <: Shape, D <: Dim]: Select[S, D, IndexOf[S, D]] with {}

    // TODO add implicit conversion like Dim.fromLongStatic so we can do "3" instead of "Idx(3)"
    /** Selects the dimension at the given index, starting from 0 */
    case class Idx[I <: Int & Singleton](i: I)
    given int[S <: Shape, I <: Int & Singleton]: Select[S, Idx[I], I] with {}

    /** Selects a specific dimension by type, for which no value might be available. */
    trait At[D <: Dim]
    object At {
      def apply[D <: Dim]: At[D] = new At {}
      def apply[D <: Dim](d: D): At[D] = new At {}
    }
    given atDim[S <: Shape, D <: Dim]: Select[S, At[D], IndexOf[S, D]] with {}

    /** Selects a dimension that's based on an earlier division (split) of another dimension */
    trait Divided[D <: Dim]
    object Divided {
      def apply[D <: Dim]: Divided[D] = new Divided {}
      def apply[D <: Dim](d: D): Divided[D] = new Divided {}
    }
    given fromDivided[S <: Shape, D <: Dim]: Select[S, Divided[D], IndexOfDivided[S, D]] with {}
  }
}
