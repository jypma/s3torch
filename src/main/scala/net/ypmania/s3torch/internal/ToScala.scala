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
import DType.*

trait ToScala[-S <: Tuple, +T <: DType] {
  type OutputType
  def apply(native: pytorch.Tensor): OutputType
}

object ToScala {
  abstract class ItemTo[V](get: pytorch.Tensor => V) {
    type OutputType = V
    def apply(native: pytorch.Tensor) = get(native)
  }

  given ItemTo[Boolean](_.item_bool) with ToScala[EmptyTuple, Bool.type] with {}
  given ItemTo[Byte](_.item_byte) with ToScala[EmptyTuple, Int8.type] with {}
  given ItemTo[Short](_.item_short) with ToScala[EmptyTuple, Int16.type] with {}
  given ItemTo[Int](_.item_int) with ToScala[EmptyTuple, Int32.type] with {}
  given ItemTo[Long](_.item_long) with ToScala[EmptyTuple, Int64.type] with {}
  given ItemTo[Float](_.item_float) with ToScala[EmptyTuple, Float32.type] with {}
  given ItemTo[Double](_.item_double) with ToScala[EmptyTuple, Float64.type] with {}

  abstract class ContiguousToArray[V: ClassTag](get: (pytorch.Tensor, Array[V]) => Unit) {
    type OutputType = Array[V]
    def apply(native: pytorch.Tensor): Array[V] = {
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

  given [D <: Dim]: ContiguousToArray[Boolean]( { (tensor, result) =>
    val buf = tensor.createBuffer[ByteBuffer]
    var i = 0
    val size = tensor.numel
    while (i < size) do {
      result(i) = buf.get(i) != 0
      i += 1
    }
  }) with ToScala[Tuple1[D], Bool.type] with {}
  given [D <: Dim]: ContiguousToArray[Byte](_.createBuffer[ByteBuffer].get(_)) with ToScala[Tuple1[D], Int8.type] with {}
  given [D <: Dim]: ContiguousToArray[Short](_.createBuffer[ShortBuffer].get(_)) with ToScala[Tuple1[D], Int16.type] with {}
  given [D <: Dim]: ContiguousToArray[Int](_.createBuffer[IntBuffer].get(_)) with ToScala[Tuple1[D], Int32.type] with {}
  given [D <: Dim]: ContiguousToArray[Long](_.createBuffer[LongBuffer].get(_)) with ToScala[Tuple1[D], Int64.type] with {}
  given [D <: Dim]: ContiguousToArray[Float](_.createBuffer[FloatBuffer].get(_)) with ToScala[Tuple1[D], Float32.type] with {}
  given [D <: Dim]: ContiguousToArray[Double](_.createBuffer[DoubleBuffer].get(_)) with ToScala[Tuple1[D], Float64.type] with {}

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
      val a = ArraySeq.unsafeWrapArray(toArray(native))
      val step = size(1).toInt // Last dim
      a.sliding(step, step).toSeq
    }
  }

  // TODO rewrite this recursively against >3 dimensions
  given [D1 <: Dim, D2 <: Dim, D3 <: Dim, V: ClassTag](using toArray: ContiguousToArray[V]): ToMultiDimSeq[(D1, D2, D3), V] with {
    def apply(native: pytorch.Tensor) = {
      val size = native.sizes.vec.get
      val a = ArraySeq.unsafeWrapArray(toArray(native))
      val step = size(2).toInt // Last dim
      a.sliding(step, step).grouped(size(1).toInt).toSeq
    }
  }

  given [S <: Tuple](using toSeq: ToMultiDimSeq[S, Boolean]): ToScala[S, Bool.type] with {
    type OutputType = MkOutputType[S, Boolean]
    def apply(native: pytorch.Tensor) = toSeq(native)
  }

  given [S <: Tuple](using toSeq: ToMultiDimSeq[S, Int]): ToScala[S, Int32.type] with {
    type OutputType = MkOutputType[S, Int]
    def apply(native: pytorch.Tensor) = toSeq(native)
  }

  given [S <: Tuple](using toSeq: ToMultiDimSeq[S, Float]): ToScala[S, Float32.type] with {
    type OutputType = MkOutputType[S, Float]
    def apply(native: pytorch.Tensor) = toSeq(native)
  }

  given [S <: Tuple](using toSeq: ToMultiDimSeq[S, Double]): ToScala[S, Float64.type] with {
    type OutputType = MkOutputType[S, Double]
    def apply(native: pytorch.Tensor) = toSeq(native)
  }

}
