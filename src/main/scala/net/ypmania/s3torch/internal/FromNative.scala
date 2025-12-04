package net.ypmania.s3torch.internal

import java.nio.ByteBuffer

import net.ypmania.s3torch.*
import net.ypmania.s3torch.Tensor.*

import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch
import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.BoolPointer
import org.bytedeco.javacpp.DoublePointer
import java.nio.DoubleBuffer

import StaticApply.ToSeq

trait FromNative[V] {
  type OutputShape <: Tuple
  type DefaultDType <: DType
  def apply[T <: DType](value: V, t: T): Tensor[OutputShape, T]
  def defaultDType: DefaultDType
}

object FromNative {
  type SupportedType = Boolean | Byte | Short | Int | Long | Float | Double

  trait FromScalar[V] extends FromNative[V] {
    type OutputShape = Scalar

    override def apply[T <: DType](value: V, dtype: T): Tensor[Scalar, T] = {
      val tensor = torch.scalar_tensor(
        toScalar(value),
        Torch.tensorOptions(dtype)
      )
      new Tensor(tensor)
    }

    def toScalar(value: V): pytorch.Scalar
  }

  trait FromSeq[V] extends FromNative[Seq[V]] {
    type OutputShape = Tuple1[Dynamic]

    override def apply[T <: DType](value: Seq[V], dtype: T): Tensor[Tuple1[Dynamic], T] = {
      val tensor = torch.from_blob(toPointer(value), Array(value.length.toLong), Torch.tensorOptions(dtype))
      new Tensor(tensor)
    }

    def toPointer(value: Seq[V]): Pointer
  }

  trait ToBool {
    type DefaultDType = Bool
    def defaultDType = bool
  }

  trait ToInt8 {
    type DefaultDType = Int8
    def defaultDType = int8
  }

  trait ToInt16 {
    type DefaultDType = Int16
    def defaultDType = int16
  }

  trait ToInt32 {
    type DefaultDType = Int32
    def defaultDType = int32
  }

  trait ToInt64 {
    type DefaultDType = Int64
    def defaultDType = int64
  }

  trait ToFloat32 {
    type DefaultDType = Float32
    def defaultDType = float32
  }

  trait ToFloat64 {
    type DefaultDType = Float64
    def defaultDType = float64
  }

  given FromScalar[Boolean] with ToBool with {
    override def toScalar(value: Boolean) = pytorch.AbstractTensor.create(value).item()
  }
  given FromSeq[Boolean] with ToBool with {
    override def toPointer(value: Seq[Boolean]) = {
      val p = new BoolPointer(value.length)
      for (idx <- 0.until(value.length)) {
        p.put(idx, value(idx))
      }
      p
    }
  }

  given FromScalar[Byte] with ToInt8 with {
    override def toScalar(value: Byte) = pytorch.Scalar(value)
  }
  given FromSeq[Byte] with ToInt8 with {
    override def toPointer(value: Seq[Byte]) = new BytePointer(ByteBuffer.wrap(value.toArray))
  }

  given FromScalar[Short] with ToInt16 with {
    override def toScalar(value: Short) = pytorch.Scalar(value)
  }

  given FromScalar[Int] with ToInt32 with {
    override def toScalar(value: Int) = pytorch.Scalar(value)
  }

  given FromScalar[Long] with ToInt64 with {
    override def toScalar(value: Long) = pytorch.Scalar(value)
  }

  given FromScalar[Float] with ToFloat32 with {
    override def toScalar(value: Float) = pytorch.Scalar(value)
  }

  given FromScalar[Double] with ToFloat64 with {
    override def toScalar(value: Double) = pytorch.Scalar(value)
  }
  given FromSeq[Double] with ToFloat64 with {
    override def toPointer(value: Seq[Double]) = new DoublePointer(DoubleBuffer.wrap(value.toArray))
  }

  trait FromTupleOps[V <: Tuple, E] {
    def toTensor[T <: DType](value: V, dtype: T): pytorch.Tensor
  }

  given [V <: Tuple, E](using toSeq: ToSeq[V, E], fromNative: FromNative[Seq[E]]): FromTupleOps[V, E] with {
    def toTensor[T <: DType](value: V, dtype: T) = fromNative.apply(toSeq.toSeq(value), dtype).native
  }

  abstract class FromTupleBase[V <: Tuple, E](using ops:FromTupleOps[V, E]) extends FromNative[V] {
    import compiletime.ops.int.ToLong
    type OutputShape = Tuple1[Dim.Static[ToLong[Tuple.Size[V]]]]
    def apply[T <: DType](value: V, t: T) = new Tensor(ops.toTensor(value, t))
  }

  given [V <: Tuple](using ops:FromTupleOps[V, Double]): FromTupleBase[V, Double] with ToFloat64 with {}
  given [V <: Tuple](using ops:FromTupleOps[V, Byte]): FromTupleBase[V, Byte] with ToInt8 with {}
  given [V <: Tuple](using ops:FromTupleOps[V, Short]): FromTupleBase[V, Short] with ToInt16 with {}
  // TODO more types, and shorten the other types above by using abstract class instead of a trait.
}
