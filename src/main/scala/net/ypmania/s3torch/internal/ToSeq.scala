package net.ypmania.s3torch.internal

import scala.compiletime.ops.long.*

trait ToSeq[-T, +A] {
  def apply(t: T): Seq[A]
}

object ToSeq {
  def toSeq[T, A](t: T)(using conv: ToSeq[T, A]): Seq[A] = conv(t)

  given [A]: ToSeq[EmptyTuple, A] with {
    override def apply(empty: EmptyTuple) = Seq.empty
  }

  given [T <: Tuple, A](using ToSeq[T, A]): ToSeq[A *: T, A] with {
    override def apply(tuple: A *: T) = tuple.head +: ToSeq.toSeq(tuple.tail)
  }

  given [T]: ToSeq[Seq[T], T] with {
    override def apply(t: Seq[T]) = t
  }
}
