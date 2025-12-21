package net.ypmania.s3torch

import scala.compiletime.ops.long.*
import org.bytedeco.pytorch

trait Index[D <: Dim, T] {
  def toNative(t: T): pytorch.TensorIndex
}

trait IndexPrio0 {
  given [D <: Dim]: Index[D, Int] with {
    def toNative(i: Int) = new pytorch.TensorIndex(i)
  }
}

object Index extends IndexPrio0 {
  case class Slice(from: Option[Int], to: Option[Int], step: Option[Int])
  object Slice:
    private def extract(index: Option[Int] | Int) = index match
      case i: Option[Int] => i
      case i: Int         => Option(i)
    def apply(
        start: Option[Int] | Int = None,
        end: Option[Int] | Int = None,
        step: Option[Int] | Int = None
    ): Slice = Slice(extract(start), extract(end), extract(step))

  given [D <: Dim]: Index[D, Slice] with {
    def toSymInt(maybeInt: Option[Int]) = maybeInt.map(l => pytorch.SymIntOptional(pytorch.SymInt(l))).orNull
    def toNative(s: Slice) = new pytorch.TensorIndex(new pytorch.Slice(toSymInt(s.from), toSymInt(s.to), toSymInt(s.step)))
  }

  // Allow a tuple with the actual dimension type, instead of just the value
  given [D <: Dim, T](using i:Index[D, T]): Index[D, (D, T)] with {
    def toNative(t: (D, T)) = i.toNative(t._2)
  }
}

trait Indices[S <: Shape, T] {
  def indexes(t: T): Seq[pytorch.TensorIndex]
  def toNative(t: T): pytorch.TensorIndexArrayRef = pytorch.TensorIndexArrayRef(new pytorch.TensorIndexVector(indexes(t)*))
}

object Indices {
  /* Allow a single Dim without tuple syntax */
  given [D <: Dim, T](using i1: Index[D, T]): Indices[Tuple1[D], T] with {
    def indexes(t: T) = Seq(i1.toNative(t))
  }

  given Indices[EmptyTuple, EmptyTuple] with {
    def indexes(t: EmptyTuple)= Seq.empty
  }

  given [D <: Dim, I, DTail <: Tuple, ITail <: Tuple](using i: Index[D, I], tail: Indices[DTail, ITail]): Indices[D *: DTail, I *: ITail] with {
    def indexes(idx: I *: ITail) = i.toNative(idx.head) +: tail.indexes(idx.tail)
  }
}
