package net.ypmania.s3torch.internal

import org.bytedeco.pytorch
import net.ypmania.s3torch.*
import java.nio.ByteBuffer
import java.nio.ShortBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.FloatBuffer
import java.nio.DoubleBuffer
import java.nio.Buffer
import scala.reflect.ClassTag
import scala.collection.immutable.ArraySeq

trait ToScala[S <: Tuple, T <: DType] {
  type OutputType
  def apply(native: pytorch.Tensor): OutputType
}

object ToScala {
  abstract class ItemTo[V](get: pytorch.Tensor => V) {
    type OutputType = V
    def apply(native: pytorch.Tensor) = get(native)
  }

  given ItemTo[Byte](_.item_byte) with ToScala[EmptyTuple, Int8] with {}
  given ItemTo[Short](_.item_short) with ToScala[EmptyTuple, Int16] with {}
  given ItemTo[Int](_.item_int) with ToScala[EmptyTuple, Int32] with {}
  given ItemTo[Long](_.item_long) with ToScala[EmptyTuple, Int64] with {}
  given ItemTo[Float](_.item_float) with ToScala[EmptyTuple, Float32] with {}
  given ItemTo[Double](_.item_double) with ToScala[EmptyTuple, Float64] with {}

  abstract class ContiguousToArray[V: ClassTag](get: (pytorch.Tensor, Array[V]) => Unit) {
    type OutputType = Array[V]
    def apply(native: pytorch.Tensor) = {
      // FIXME: Need to move to CPU, and to Strided format, if not already.
      val size = native.numel()
      if (size > Int.MaxValue) {
        throw new IllegalStateException("Tensor too big to fit in Java array")
      }
      val a = new Array[V](size.toInt)
      if (size > 0) {
        get(native.contiguous(), a)
      }
      a
    }
  }

  given [D <: Dim]: ContiguousToArray[Byte](_.createBuffer[ByteBuffer].get(_)) with ToScala[Tuple1[D], Int8] with {}
  given [D <: Dim]: ContiguousToArray[Short](_.createBuffer[ShortBuffer].get(_)) with ToScala[Tuple1[D], Int16] with {}
  given [D <: Dim]: ContiguousToArray[Int](_.createBuffer[IntBuffer].get(_)) with ToScala[Tuple1[D], Int32] with {}
  given [D <: Dim]: ContiguousToArray[Long](_.createBuffer[LongBuffer].get(_)) with ToScala[Tuple1[D], Int64] with {}
  given [D <: Dim]: ContiguousToArray[Float](_.createBuffer[FloatBuffer].get(_)) with ToScala[Tuple1[D], Float32] with {}
  given [D <: Dim]: ContiguousToArray[Double](_.createBuffer[DoubleBuffer].get(_)) with ToScala[Tuple1[D], Float64] with {}

  type MkOutputType[S <: Tuple, ElemType] = S match {
    case EmptyTuple => ElemType
    case Tuple1[dim] => Seq[ElemType]
    case dim *: tail => Seq[MkOutputType[tail, ElemType]]
  }

  abstract class ToMultiDimSeq[S <: Tuple, V: ClassTag] {
    type OutputType = MkOutputType[S, V]
    def apply(native: pytorch.Tensor): OutputType
  }

  given [V: ClassTag](using itemTo: ItemTo[V]): ToMultiDimSeq[EmptyTuple, V] with {
    def apply(native: pytorch.Tensor) = itemTo(native)
  }

  given [D1 <: Dim, V: ClassTag](using toArray: ContiguousToArray[V]): ToMultiDimSeq[Tuple1[D1], V] with {
    def apply(native: pytorch.Tensor) = ArraySeq.unsafeWrapArray(toArray(native))
  }

  given [D1 <: Dim, D2 <: Dim, V: ClassTag](using toArray: ContiguousToArray[V]): ToMultiDimSeq[(D1, D2), V] with {
    def apply(native: pytorch.Tensor) = {
      val size = native.sizes.vec.get
      val a = toArray(native).toSeq
      val step = size(1).toInt
      a.sliding(step, step).toSeq
    }
  }

  given [D1 <: Dim, D2 <: Dim](using toSeq: ToMultiDimSeq[(D1, D2), Int]): ToScala[(D1, D2), Int32] with {
    type OutputType = MkOutputType[(D1, D2), Int]
    def apply(native: pytorch.Tensor) = toSeq(native)
  }
}
