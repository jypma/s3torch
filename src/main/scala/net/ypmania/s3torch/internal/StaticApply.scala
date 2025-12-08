package net.ypmania.s3torch.internal

import scala.compiletime.ops.long.*

object StaticApply {
  trait ToSeq[-T, +A] {
    def toSeq(t: T): Seq[A]
  }

  object ToSeq {
    def toSeq[T, A](t: T)(using conv: ToSeq[T, A]): Seq[A] = conv.toSeq(t)
  }

  given [A]: ToSeq[EmptyTuple, A] with {
    override def toSeq(empty: EmptyTuple) = Seq.empty
  }

  given [T <: Tuple, A](using ToSeq[T, A]): ToSeq[A *: T, A] with {
    override def toSeq(tuple: A *: T) = tuple.head +: ToSeq.toSeq(tuple.tail)
  }

  given id[T]: ToSeq[Seq[T], T] with {
    def toSeq(t: Seq[T]) = t
  }
}
