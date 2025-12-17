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
  // TODO match Dim.Static explicitly with a Int & Singleton. We'll have to introduce a Conversion[Int & Singleton, StaticIndex] and then a given for StaticIndex.

  // Allow a tuple with the actual dimension type, instead of just the value
  given [D <: Dim, T](using i:Index[D, T]): Index[D, (D, T)] with {
    def toNative(t: (D, T)) = i.toNative(t._2)
  }
}

trait Indices[S <: Shape, T] {
  def toNative(t: T): pytorch.TensorIndexArrayRef
}

object Indices {
  // We explicitly don't define EmptyTuple here, since setting a single value through .value = would be nicer syntax.
  given [D <: Dim, T](using i1: Index[D, T]): Indices[Tuple1[D], T] with {
    def toNative(t: T) = pytorch.TensorIndexArrayRef(new pytorch.TensorIndexVector(i1.toNative(t)))
  }
  given [D1 <: Dim, D2 <: Dim, T1](using i1: Index[D1, T1]): Indices[(D1, D2), T1] with {
    def toNative(t: T1) = pytorch.TensorIndexArrayRef(new pytorch.TensorIndexVector(i1.toNative(t)))
  }
  // TODO rewrite as recursion
  given [D1 <: Dim, D2 <: Dim, T1, T2](using i1: Index[D1, T1], i2: Index[D2, T2]): Indices[(D1, D2), (T1, T2)] with {
    def toNative(t: (T1, T2)) = pytorch.TensorIndexArrayRef(new pytorch.TensorIndexVector(i1.toNative(t._1), i2.toNative(t._2)))
  }
}
