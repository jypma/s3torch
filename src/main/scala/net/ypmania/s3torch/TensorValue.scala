package net.ypmania.s3torch

import net.ypmania.s3torch.Dim.UnRef
import scala.reflect.ClassTag

/**
  * Carries an on-heap, immutable tensor with any Scala value, but a Shape that's compatible with Tensor.
  */
class TensorValue[S <: Tuple, +T](private val contents: Seq[T]) extends Iterable[T] {
  override def iterator: Iterator[T] = contents.iterator

  def map[U: ClassTag](fn: T => U): TensorValue[S, U] = new TensorValue(contents.map(fn))
}

object TensorValue {
  def arangeOf[D <: Dim](dim: D)(using u:UnRef[D]): TensorValue[Tuple1[u.Out], Long] = new TensorValue((0L.until(dim.size)))

  /** Creates a TensorValue from an Iterable, and an expected Dim. The iterable must contain exactly [Dim] elements. */
  def apply[D <: Dim, T](elements: Iterable[T], dim: D)(using u:UnRef[D]): TensorValue[Tuple1[u.Out], T] = {
    if (dim.size != elements.size) {
      throw new IllegalArgumentException(s"Expected ${dim.size} elements, but got ${elements.size}")
    }
    new TensorValue(elements.toVector)
  }

  // ---- Methods on TensorValue with 1 dimension ---
  extension[T: ClassTag, D1 <: Dim](t: TensorValue[Tuple1[D1], T]) {
    def toArray: Array[T] = t.contents.toArray
  }

}
