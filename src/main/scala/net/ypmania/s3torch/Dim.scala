package net.ypmania.s3torch

import  scala.compiletime.ops.long.*

trait Dim {
  def size: Long
}
trait DimLowPriorityGivens {
  given fromLongDynamic[L <: Long]: Conversion[L, Dim.Dynamic] with {
    def apply(l: L) = Dim.Dynamic(l)
  }
}
object Dim extends DimLowPriorityGivens {
  /** A dimension known at compile time */
  abstract class Static[S <: Long](using ValueOf[S]) extends Dim {
    type Size = S
    def size = valueOf[S]
  }
  object Static {
    def apply[L <: Long & Singleton](l: L)(using ValueOf[L]) = new Static[L] {}
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

  // TODO ---------------- move division stuff internals to different file under internal ----------------

  type IsDivisibleByLong[A <: Long, B <: Long] = A % B match {
      case 0L => true
      case _ => false
  }

  type IsDivisibleBy[D, L <: Long] <: Boolean = D match {
    case Long => IsDivisibleByLong[D, L]
    case Dim.Static[v] => IsDivisibleByLong[v, L]
    case _ => false
  }

  trait DivisibleBy[+D, +L <: Long] {
    type Res <: Long
  }

  type StaticSize[D] <: Long = D match {
    case Dim.Static[size] => size
  }

  object DivisibleBy {
    given fromNums[A <: Long, B <: Long](using A % B =:= 0L): DivisibleBy[A, B] with {}
    given fromNum[A <: Long, D <: Dim.Static[A], B <: Long](using DivisibleBy[A, B]): DivisibleBy[D, B] with {}
    given fromDim[D, B <: Long](using StaticSize[D] % B =:= 0L): DivisibleBy[D, B] with {}
  }
  infix type |/[+D, +L <: Long] = DivisibleBy[D, L]

  type DividedBy[D, L <: Long] <: Long = D match {
    case Long => scala.compiletime.ops.long./[D, L]
    case Dim.Static[v] => scala.compiletime.ops.long./[v, L]
  }
  abstract class DividedDim[D, L, R <: Long](using ValueOf[R]) extends Dim {
    type Orig = D
    type Divisor = L
    override def size = valueOf[R]
  }
  infix type /[D, L <: Long] = DividedDim[D, L, DividedBy[D, L]]
}
