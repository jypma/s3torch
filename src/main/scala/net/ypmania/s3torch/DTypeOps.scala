package net.ypmania.s3torch

import net.ypmania.s3torch.DType.BFloat16
import net.ypmania.s3torch.DType.Bool
import net.ypmania.s3torch.DType.Float16
import net.ypmania.s3torch.DType.Float32
import net.ypmania.s3torch.DType.Float64
import net.ypmania.s3torch.DType.Int16
import net.ypmania.s3torch.DType.Int32
import net.ypmania.s3torch.DType.Int64
import net.ypmania.s3torch.DType.Int8
import org.bytedeco.javacpp.BoolPointer
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.DoublePointer
import org.bytedeco.javacpp.FloatPointer
import org.bytedeco.javacpp.IntPointer
import org.bytedeco.javacpp.LongPointer
import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.ShortPointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch

import java.nio.ByteBuffer
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer
import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag

// NOT +T, because sub-dtypes will have different ScalaType, typically.
/** Contains sequence-based operations to convert Scala instances to/from a libtorch DType */
trait DTypeOps[T <: DType, S] extends DTypeOps.Scalar[T, S]{
  type ScalaType = S
  def dType: T

  def toPointer(v: Seq[ScalaType]): Pointer

  def copyToArray(t: pytorch.Tensor, a: Array[ScalaType]): Unit
  def toArray(t: pytorch.Tensor)(using ClassTag[ScalaType]): Array[ScalaType] = {
    // This assumes the tensor is on CPU, and in Strided format. Tensor.scala makes sure of that.
    val size = t.numel()
    if (size > Int.MaxValue) {
      throw new IllegalStateException("Tensor too big to fit in Java array")
    }
    val a = new Array[ScalaType](size.toInt)
    if (size > 0) {
      copyToArray(t.contiguous(), a)
    }
    a
  }
  def toSeq(t: pytorch.Tensor)(using ClassTag[ScalaType]): Seq[ScalaType] = ArraySeq.unsafeWrapArray(toArray(t))
  def toSeq2D(t: pytorch.Tensor)(using ClassTag[ScalaType]): Seq[Seq[ScalaType]] = {
    val size = t.sizes.vec.get
    val step = size(1).toInt // Last dim
    toSeq(t).sliding(step, step).toSeq
  }
  def toSeq3D(t: pytorch.Tensor)(using ClassTag[ScalaType]): Seq[Seq[Seq[ScalaType]]] = {
    val size = t.sizes.vec.get
    val step = size(2).toInt // Last dim
    toSeq(t).sliding(step, step).grouped(size(1).toInt).toSeq
  }
}

object DTypeOps {
  /** Contains scalar-based operations to convert Scala instances to/from a libtorch DType */
  trait Scalar[T <: DType, S] {
    def fromScalar(v: S): pytorch.Scalar
    def toScalar(t: pytorch.Tensor): S

    def toValue(double: Double): S
    def toDouble(s: S): Double
    def toValue(long: Long): S
    def toLong(s: S): Long
  }

  given bool: DTypeOps[Bool, Boolean] with Scalar[Bool, Boolean] {
    def dType = DType.bool
    type ScalaType = Boolean

    override def fromScalar(v: Boolean): pytorch.Scalar = pytorch.AbstractTensor.create(v).item()
    override def toPointer(v: Seq[ScalaType]) = {
      val p = new BoolPointer(v.length)
      for (idx <- 0.until(v.length)) {
        p.put(idx, v(idx))
      }
      p

    }
    override def toScalar(t: pytorch.Tensor) = t.item_bool
    override def copyToArray(t: pytorch.Tensor, a: Array[Boolean]) = {
      val buf = t.createBuffer[ByteBuffer]
      var i = 0
      val size = t.numel
      while (i < size) do {
        a(i) = buf.get(i) != 0
        i += 1
      }
    }

    override def toValue(double: Double) = double != 0.0
    override def toDouble(s: Boolean) = if (s) 1.0 else 0.0
    override def toValue(long: Long) = long != 0L
    override def toLong(s: Boolean) = if (s) 1L else 0L
  }

  given float16: DTypeOps[Float16, Float] {
    def dType = DType.float16
    override def fromScalar(v: ScalaType): pytorch.Scalar = pytorch.Scalar(pytorch.Half(v))
    override def toPointer(v: Seq[ScalaType]) = throw new UnsupportedOperationException("No defined syntax to create a multivalue Float16")
    override def toScalar(t: pytorch.Tensor) = t.item_float
    override def copyToArray(t: pytorch.Tensor, a: Array[ScalaType]) = {
      val shorts = t.createBuffer[ShortBuffer]
      var i = 0
      while (i < shorts.limit()) {
        a(i) = torch.fp16_ieee_to_fp32_value(shorts.get(i))
        i += 1
      }
    }

    override def toValue(double: Double) = double.toFloat
    override def toDouble(s: ScalaType) = s.toDouble
    override def toValue(long: Long) = long.toFloat
    override def toLong(s: ScalaType) = s.toLong
  }

  given bfloat16: DTypeOps[BFloat16, Float] {
    def dType = DType.bfloat16
    override def fromScalar(v: ScalaType): pytorch.Scalar = pytorch.Scalar(pytorch.BFloat16(v))
    override def toPointer(v: Seq[ScalaType]) = throw new UnsupportedOperationException("No defined syntax to create a multivalue BFloat16")
    override def toScalar(t: pytorch.Tensor) = t.item_float
    override def copyToArray(t: pytorch.Tensor, a: Array[ScalaType]) = {
      val shorts = t.createBuffer[ShortBuffer]
      var i = 0
      while (i < shorts.limit()) {
        a(i) = torch.fp32_from_bits(shorts.get(i).toInt << 16)
        i += 1
      }
    }

    override def toValue(double: Double) = double.toFloat
    override def toDouble(s: ScalaType) = s.toDouble
    override def toValue(long: Long) = long.toFloat
    override def toLong(s: ScalaType) = s.toLong
  }

  given float32: DTypeOps[Float32, Float] {
    def dType = DType.float32
    override def fromScalar(v: ScalaType): pytorch.Scalar = pytorch.Scalar(v)
    override def toPointer(v: Seq[ScalaType]) = new FloatPointer(FloatBuffer.wrap(v.toArray))
    override def toScalar(t: pytorch.Tensor) = t.item_float
    override def copyToArray(t: pytorch.Tensor, a: Array[ScalaType]) = t.createBuffer[FloatBuffer].get(a)

    override def toValue(double: Double) = double.toFloat
    override def toDouble(s: ScalaType) = s.toDouble
    override def toValue(long: Long) = long.toFloat
    override def toLong(s: ScalaType) = s.toLong
  }

  given float64: DTypeOps[Float64, Double] {
    def dType = DType.float64
    override def fromScalar(v: ScalaType): pytorch.Scalar = pytorch.Scalar(v)
    override def toPointer(v: Seq[ScalaType]) = new DoublePointer(DoubleBuffer.wrap(v.toArray))
    override def toScalar(t: pytorch.Tensor) = t.item_double
    override def copyToArray(t: pytorch.Tensor, a: Array[ScalaType]) = t.createBuffer[DoubleBuffer].get(a)

    override def toValue(double: Double) = double
    override def toDouble(s: ScalaType) = s
    override def toValue(long: Long) = long.toFloat
    override def toLong(s: ScalaType) = s.toLong
  }

  /** For "Floaty" DTypes, we allow double literals to be used instead of the known type T. */
  given scalarFromDouble[T <: DType.Floaty, V](using ops: Scalar[T, V]): Scalar[T, Double] with {
    override def fromScalar(v: Double): pytorch.Scalar = ops.fromScalar(ops.toValue(v))
    override def toScalar(t: pytorch.Tensor) = ops.toDouble(ops.toScalar(t))

    override def toValue(double: Double) = double
    override def toDouble(value: Double) = value
    override def toValue(long: Long) = long.toDouble
    override def toLong(s: Double) = s.toLong
  }

  /** For any Scalar DType, we allow Long to be used in certain places
    * to create scalars. But this is done explicitly by instantiating
    * this class (not as a given). */
  class ScalarFromLong[T <: DType, V](ops: Scalar[T, V]) extends Scalar[T, Long] {
    override def fromScalar(v: Long): pytorch.Scalar = ops.fromScalar(ops.toValue(v))
    override def toScalar(t: pytorch.Tensor) = ops.toLong(ops.toScalar(t))

    override def toValue(double: Double) = double.toLong
    override def toDouble(value: Long) = value.toDouble
    override def toValue(long: Long) = long
    override def toLong(s: Long) = s
   }

  // The int types are generic, to allow opaque types on top of them (for type-safe token-based tensors)
  class Int8Ops[T <: Int8, S <: Byte](val dType: T) extends DTypeOps[T, S] {
    override def fromScalar(v: ScalaType): pytorch.Scalar = pytorch.Scalar(v)
    override def toPointer(v: Seq[ScalaType]) = new BytePointer(ByteBuffer.wrap(v.toArray))
    override def toScalar(t: pytorch.Tensor) = t.item_byte.asInstanceOf[S]
    override def copyToArray(t: pytorch.Tensor, a: Array[S]) = t.createBuffer[ByteBuffer].get(a.asInstanceOf[Array[Byte]])

    override def toValue(double: Double) = double.toByte.asInstanceOf[ScalaType]
    override def toDouble(s: ScalaType) = s.toDouble
    override def toValue(long: Long) = long.toByte.asInstanceOf[ScalaType]
    override def toLong(s: ScalaType) = s.toLong

  }
  given int8: DTypeOps[Int8, Byte] = new Int8Ops[Int8, Byte](DType.int8)

  class Int16Ops[T <: Int16, S <: Short](val dType: T) extends DTypeOps[T, S] {
    override def fromScalar(v: ScalaType): pytorch.Scalar = pytorch.Scalar(v)
    override def toPointer(v: Seq[ScalaType]) = new ShortPointer(ShortBuffer.wrap(v.toArray))
    override def toScalar(t: pytorch.Tensor) = t.item_short.asInstanceOf[S]
    override def copyToArray(t: pytorch.Tensor, a: Array[S]) = t.createBuffer[ShortBuffer].get(a.asInstanceOf[Array[Short]])

    override def toValue(double: Double) = double.toShort.asInstanceOf[ScalaType]
    override def toDouble(s: ScalaType) = s.toDouble
    override def toValue(long: Long) = long.toByte.asInstanceOf[ScalaType]
    override def toLong(s: ScalaType) = s.toLong
  }
  given int16: DTypeOps[Int16, Short] = new Int16Ops[Int16, Short](DType.int16)

  class Int32Ops[T <: Int32, S <: Int](val dType: T) extends DTypeOps[T, S] {
    override def fromScalar(v: S): pytorch.Scalar = pytorch.Scalar(v)
    override def toPointer(v: Seq[S]) = new IntPointer(IntBuffer.wrap(v.toArray))
    override def toScalar(t: pytorch.Tensor) = t.item_int.asInstanceOf[S]
    override def copyToArray(t: pytorch.Tensor, a: Array[S]) = t.createBuffer[IntBuffer].get(a.asInstanceOf[Array[Int]])

    override def toValue(double: Double) = double.toInt.asInstanceOf[ScalaType]
    override def toDouble(s: ScalaType) = s.toDouble
    override def toValue(long: Long) = long.toByte.asInstanceOf[ScalaType]
    override def toLong(s: ScalaType) = s.toLong
  }
  given int32: DTypeOps[Int32, Int] = new Int32Ops[Int32, Int](DType.int32)

  class Int64Ops[T <: Int64, S <: Long](val dType: T) extends DTypeOps[T, S] {
    override def fromScalar(v: ScalaType): pytorch.Scalar = pytorch.Scalar(v)
    override def toPointer(v: Seq[ScalaType]) = new LongPointer(LongBuffer.wrap(v.toArray))
    override def toScalar(t: pytorch.Tensor) = t.item_long.asInstanceOf[S]
    override def copyToArray(t: pytorch.Tensor, a: Array[S]) = t.createBuffer[LongBuffer].get(a.asInstanceOf[Array[Long]])

    override def toValue(double: Double) = double.toLong.asInstanceOf[ScalaType]
    override def toDouble(s: ScalaType) = s.toDouble
    override def toValue(long: Long) = long.asInstanceOf[ScalaType]
    override def toLong(s: ScalaType) = s
  }
  given int64: DTypeOps[Int64, Long] = new Int64Ops[Int64, Long](DType.int64)

  // TODO introduce an actual Token DType, that combines a Dim (of size VocabSize) with a DType that indexes that dim.
}
