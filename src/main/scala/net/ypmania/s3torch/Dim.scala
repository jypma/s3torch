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

  /** A dimension not known until runtime */
  class Dynamic(_size: Long) extends Dim {
    override def size = _size
  }

  /** A reference made to an unknown dimension (typically used as a type parameter to generic building blocks) */
  case class Ref[D <: Dim](ref: D) extends Dim {
    override def size = ref.size
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

  type Max[D1 <: Dim, D2 <: Dim] <: Dim = D1 match {
    case D2 => D1 // Same type => pick any
    case _ => (D1, D2) match {
      case (One, _) => D2 // Either Dim is one => pick the other
      case (_, One) => D1
      case (Static[s1], Static[s2]) => (s1 > s2) match {
        case true => D1 // If static, pick whichever is bigger
        case false => D2
      }
      case _ => Dynamic // Fallback, we don't know statically which one is bigger
    }
  }

  trait DimArg[D <: Dim] {
    type Out <: Dim
    def apply(d: D): Out
  }
  trait DimArgPrio0 {
    given mkRef[D <: Dim]: DimArg[D] with {
      type Out = Ref[D]
      def apply(d: D) = Ref(d)
    }
  }
  object DimArg extends DimArgPrio0 {
    given asStatic[L <: Long, D <: Static[L]]: DimArg[D] with {
      type Out = D
      def apply(d: D) = d
    }
    given asDynamic[D <: Dim.Dynamic]: DimArg[D] with {
      type Out = D
      def apply(d: D) = d
    }
    given asRef[D <: Dim]: DimArg[Ref[D]] with {
      type Out = Ref[D]
      def apply(d: Ref[D]) = d
    }
  }
}
