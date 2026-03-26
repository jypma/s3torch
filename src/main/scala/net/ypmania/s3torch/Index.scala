package net.ypmania.s3torch

import org.bytedeco.pytorch
import scala.annotation.implicitNotFound
import scala.compiletime.ops.int.ToLong
import scala.compiletime.ops.int.+
import scala.compiletime.ops.long.<
import scala.util.NotGiven

/** A way to specify an index, or range of indexes, into a given dimension. */
trait Index {
  def toNative: pytorch.TensorIndex
}

trait IndexPrio0 {
  given fromIntDynamic: Conversion[Int, Index.At] with {
    def apply(i: Int) = Index.At(i)
  }
  given validAnyDim[D <: Dim, I <: Int & Singleton](using NotGiven[Index.Invalid[D, Index.Idx[I]]]): Index.Valid[D, Index.Idx[I]] with {
    type Apply = EmptyTuple
  }
}

object Index extends IndexPrio0 {
  @implicitNotFound("Index is not valid for this tensor. Perhaps it is out of bounds?")
  trait Valid[D <: Dim, I <: Index] {
    type Apply <: Tuple
  }

  trait Invalid[D <: Dim, I <: Index]
  given invalidStaticDim[L <: Long, D <: Dim.Static[L], I <: Int & Singleton](using ToLong[I] < L =:= false): Invalid[D, Idx[I]] with {}

  /** A statically known index into a dimension */
  case class Idx[I <: Int & Singleton](value: I) extends Index {
    def toNative = new pytorch.TensorIndex(value)
  }
  given fromIntStatic[I <: Int & Singleton](using ValueOf[I], ValueOf[I + 0]): Conversion[I, Idx[I]] with {
    def apply(i: I) = Idx(i)
  }

  // TODO consider introducing the type value (Int & Singleton) |<= D2
  /** An index only known at runtime, which is not bounds-checked. */
  case class At(value: Int) extends Index {
    def toNative = new pytorch.TensorIndex(value)
  }
  given [D <: Dim]: Valid[D, At] with {
    type Apply = EmptyTuple
  }

  // FIXME Align Select.First and Index.First
  /** Selects the first element in that dimension */
  case object First extends Index {
    def toNative = new pytorch.TensorIndex(0)
  }
  given [D <: Dim]: Valid[D, First.type] with {
    type Apply = EmptyTuple
  }

  /** Selects the last element in that dimension */
  case object Last extends Index {
    def toNative = new pytorch.TensorIndex(-1)
  }
  given [D <: Dim]: Valid[D, Last.type] with {
    type Apply = EmptyTuple
  }

  /** Selects the full dimension */
  case object All extends Index {
    def toNative = new pytorch.TensorIndex(new pytorch.Slice(new pytorch.SymIntOptional, new pytorch.SymIntOptional, new pytorch.SymIntOptional))
  }
  given [D <: Dim]: Valid[D, All.type] with {
    type Apply = Tuple1[D]
  }

  /** Selects elements up to the size of [D] (which must be smaller than or equal than the current dimension in the shape) */
  // TODO consider introducing the type D1 |<= D2 to prove that D1 is smaller than or equal to D2
  case class Take[D](size: Int) extends Index {
    def toNative = new pytorch.TensorIndex(new pytorch.Slice(toSymInt(None), toSymInt(Some(size)), toSymInt(None)))
  }
  case object Take {
    def apply[D <: Dim](dim: Dim.Ref[D]): Take[D] = Take(dim.size.toInt)
    def apply[D <: Dim](dim: D): Take[D] = Take(dim.size.toInt)
  }
  given givtake[D <: Dim, O]: Valid[D, Take[O]] with {
    type Apply = Tuple1[O]
  }

  /** Selects a subset of a dimension. */
  case class Slice(from: Option[Int], to: Option[Int], step: Option[Int]) extends Index {
    def toNative = new pytorch.TensorIndex(new pytorch.Slice(toSymInt(from), toSymInt(to), toSymInt(step)))
  }
  object Slice:
    private def extract(index: Option[Int] | Int) = index match
      case i: Option[Int] => i
      case i: Int         => Option(i)
    def apply(
        start: Option[Int] | Int = None,
        end: Option[Int] | Int = None,
        step: Option[Int] | Int = None
    ): Slice = Slice(extract(start), extract(end), extract(step))

  given [D <: Dim]: Valid[D, Slice] with {
    type Apply = Tuple1[Dim]
  }

  // Allow a tuple with the actual dimension type, instead of just the value
  given fromTuple[D <: Dim, T <: Index]: Conversion[(D, T), T] with {
    def apply(t: (D, T)) = t._2
  }
  // Can't use singleton ints inside tuples, so we can't bounds check here.
  given fromTupleInt[D <: Dim]: Conversion[(D, Int), At] with {
    def apply(t: (D, Int)) = At(t._2)
  }

  private def toSymInt(maybeInt: Option[Int]) = maybeInt.map(l => pytorch.SymIntOptional(pytorch.SymInt(l))).orNull
}
