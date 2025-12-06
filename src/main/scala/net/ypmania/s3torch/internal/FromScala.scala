package net.ypmania.s3torch.internal

import java.nio.ByteBuffer
import java.nio.ShortBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.FloatBuffer
import java.nio.DoubleBuffer

import net.ypmania.s3torch.*
import net.ypmania.s3torch.Tensor.*

import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch
import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.ShortPointer
import org.bytedeco.javacpp.IntPointer
import org.bytedeco.javacpp.LongPointer
import org.bytedeco.javacpp.BoolPointer
import org.bytedeco.javacpp.FloatPointer
import org.bytedeco.javacpp.DoublePointer

import StaticApply.ToSeq

trait FromScala[V] {
  type OutputShape <: Tuple
  type DefaultDType <: DType
  def apply[T <: DType](value: V, t: T): Tensor[OutputShape, T]
  def defaultDType: DefaultDType
}

object FromScala {
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

  abstract class FromScalar[V](toScalar: V => pytorch.Scalar) extends FromScala[V] {
    type OutputShape = Scalar

    override def apply[T <: DType](value: V, dtype: T): Tensor[Scalar, T] = {
      val tensor = torch.scalar_tensor(
        toScalar(value),
        Torch.tensorOptions(dtype)
      )
      new Tensor(tensor)
    }
  }

  given FromScalar[Byte](pytorch.Scalar(_)) with ToInt8 with {}
  given FromScalar[Short](pytorch.Scalar(_)) with ToInt16 with {}
  given FromScalar[Int](pytorch.Scalar(_)) with ToInt32 with {}
  given FromScalar[Long](pytorch.Scalar(_)) with ToInt64 with {}
  given FromScalar[Float](pytorch.Scalar(_)) with ToFloat32 with {}
  given FromScalar[Double](pytorch.Scalar(_)) with ToFloat64 with {}
  given FromScalar[Boolean](value => pytorch.AbstractTensor.create(value).item()) with ToBool with {}

  abstract class FromSeq[V](toPointer: Seq[V] => Pointer) extends FromScala[Seq[V]] {
    type OutputShape = Tuple1[Dim.Dynamic]

    override def apply[T <: DType](value: Seq[V], dtype: T): Tensor[Tuple1[Dim.Dynamic], T] = {
      val tensor = torch.from_blob(toPointer(value), Array(value.length.toLong), Torch.tensorOptions(dtype))
      new Tensor(tensor)
    }
  }

  given FromSeq[Byte](v => new BytePointer(ByteBuffer.wrap(v.toArray))) with ToInt8 with {}
  given FromSeq[Short](v => new ShortPointer(ShortBuffer.wrap(v.toArray))) with ToInt16 with {}
  given FromSeq[Int](v => new IntPointer(IntBuffer.wrap(v.toArray))) with ToInt32 with {}
  given FromSeq[Long](v => new LongPointer(LongBuffer.wrap(v.toArray))) with ToInt64 with {}
  given FromSeq[Float](v => new FloatPointer(FloatBuffer.wrap(v.toArray))) with ToFloat32 with {}
  given FromSeq[Double](v => new DoublePointer(DoubleBuffer.wrap(v.toArray))) with ToFloat64 with {}
  given FromSeq[Boolean](value => {
    val p = new BoolPointer(value.length)
    for (idx <- 0.until(value.length)) {
      p.put(idx, value(idx))
    }
    p
  }) with ToBool with {}

  trait FromTupleOps[V <: Tuple, E] {
    def toTensor[T <: DType](value: V, dtype: T): pytorch.Tensor
  }

  given [V <: Tuple, E](using toSeq: ToSeq[V, E], fromScala: FromScala[Seq[E]]): FromTupleOps[V, E] with {
    def toTensor[T <: DType](value: V, dtype: T) = fromScala(toSeq.toSeq(value), dtype).native
  }

  abstract class FromTupleBase[V <: Tuple, E](using ops:FromTupleOps[V, E]) extends FromScala[V] {
    import compiletime.ops.int.ToLong
    type OutputShape = Tuple1[Dim.Static[ToLong[Tuple.Size[V]]]]
    def apply[T <: DType](value: V, t: T) = new Tensor(ops.toTensor(value, t))
  }

  given [V <: Tuple](using ops:FromTupleOps[V, Byte]): FromTupleBase[V, Byte] with ToInt8 with {}
  given [V <: Tuple](using ops:FromTupleOps[V, Short]): FromTupleBase[V, Short] with ToInt16 with {}
  given [V <: Tuple](using ops:FromTupleOps[V, Int]): FromTupleBase[V, Int] with ToInt32 with {}
  given [V <: Tuple](using ops:FromTupleOps[V, Long]): FromTupleBase[V, Long] with ToInt64 with {}
  given [V <: Tuple](using ops:FromTupleOps[V, Float]): FromTupleBase[V, Float] with ToFloat32 with {}
  given [V <: Tuple](using ops:FromTupleOps[V, Double]): FromTupleBase[V, Double] with ToFloat64 with {}
}
