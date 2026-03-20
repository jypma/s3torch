package net.ypmania.s3torch

import scala.compiletime.ops.long._
import scala.compiletime.ops.int.ToLong

trait Dim extends Ordered[Dim] {
  def size: Long
  override def compare(that: Dim) = java.lang.Long.compare(this.size, that.size)
}
trait DimLowPriorityGivens {
  given fromLongDynamic[L <: Long]: Conversion[L, Dim.Dynamic] with {
    def apply(l: L) = Dim.Dynamic(l)
  }
}
object Dim extends DimLowPriorityGivens {
  def apply(size: Long): Dynamic = new Dynamic(size)

  /** A dimension known at compile time */
  abstract class Static[S <: Long](using ValueOf[S]) extends Dim {
    type Size = S
    def size = valueOf[S]
  }
  object Static {
    def apply[L <: Long & Singleton](l: L)(using ValueOf[L]) = new Static[L] {}
    def apply[I <: Int & Singleton](l: I) = new Static[ToLong[I]](using ValueOf(l.toLong.asInstanceOf[ToLong[I]])) {}
  }

  /** A dimension not known until runtime */
  class Dynamic(_size: Long) extends Dim {
    override def size = _size
  }

  // The "+ 0L" hack here is needed, since scala 3.7.4 otherwise will allow Long variables to match here, even though
  // their compile-time value is unknown.
  given fromLongStatic[L <: Long & Singleton](using ValueOf[L], ValueOf[L + 0L]): Conversion[L, Static[L]] with {
    def apply(l: L) = new Static[L] {
      override def size = valueOf[L]
    }
  }

  /** A dimension known to be 1 at compile time */
  type One = Static[1L]

  /** The statically-known size of a Dim */
  type StaticSize[D] <: Long = D match {
    case Dim.Static[size] => size
  }

  /** A Dim that is the result of multiplying two other Dims */
  trait ProductDim[A <: Dim, B <: Dim] extends Dim
  infix type *[A <: Dim, B <: Dim] = ProductDim[A, B]

  /** Proof that D is divisible by L */
  trait DivisibleBy[+D <: Dim, +L <: Dim] {}
  object DivisibleBy {
    given [A <: Dim, B <: Dim](using StaticSize[A] % StaticSize[B] =:= 0L): DivisibleBy[A, B] with {}
  }
  infix type |/[+D <: Dim, +L <: Dim] = DivisibleBy[D, L]

  /** A Dim that is the result of dividing two other Dims */
  trait DividedDim[D <: Dim, L <: Dim] extends Dim {}
  infix type /[D <: Dim, L <: Dim] = DividedDim[D, L]
}
