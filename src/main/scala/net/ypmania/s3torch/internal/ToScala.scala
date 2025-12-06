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

trait ToScala[S <: Tuple, T <: DType] {
  type OutputType
  def apply(native: pytorch.Tensor): OutputType
}

object ToScala {
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

  abstract class To[V](get: pytorch.Tensor => V) {
    type OutputType = V
    def apply(native: pytorch.Tensor) = get(native)
  }

  given To[Byte](_.item_byte) with ToScala[EmptyTuple, Int8] with {}
  given To[Short](_.item_short) with ToScala[EmptyTuple, Int16] with {}
  given To[Int](_.item_int) with ToScala[EmptyTuple, Int32] with {}
  given To[Long](_.item_long) with ToScala[EmptyTuple, Int64] with {}
  given To[Float](_.item_float) with ToScala[EmptyTuple, Float32] with {}
  given To[Double](_.item_double) with ToScala[EmptyTuple, Float64] with {}
}
