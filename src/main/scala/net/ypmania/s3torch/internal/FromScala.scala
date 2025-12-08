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

import compiletime.ops.int.ToLong

trait FromScala[V] {
  type OutputShape <: Tuple
  type DefaultDType <: DType
  def apply[T <: DType](value: V, t: T): Tensor[OutputShape, T]
  def defaultDType: DefaultDType
}

object FromScala {
  // We need to explicitly extend these traits directly for each given. If we pull them in using givens, the types don't resolve.
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

  type ToShape[V] <: Tuple = V match {
    case Seq[elem] => Dim.Dynamic *: ToShape[elem]
    case Tuple => Dim.Static[ToLong[Tuple.Size[V]]] *: ToShape[Tuple.Union[V]]
    case _ => EmptyTuple
  }

  abstract class FromSeq[S, V](toPointer: Seq[V] => Pointer)(using toSeq: ToSeq[S, V]) extends FromScala[S] {
    type OutputShape = ToShape[S]

    override def apply[T <: DType](value: S, dtype: T): Tensor[OutputShape, T] = {
      val seq = toSeq(value)
      val tensor = torch.from_blob(toPointer(seq), Array(seq.length.toLong), Torch.tensorOptions(dtype))
      new Tensor(tensor)
    }
  }

  given [S](using ToSeq[S, Byte]): FromSeq[S, Byte](v => new BytePointer(ByteBuffer.wrap(v.toArray))) with ToInt8 with {}
  given [S](using ToSeq[S, Short]): FromSeq[S, Short](v => new ShortPointer(ShortBuffer.wrap(v.toArray))) with ToInt16 with {}
  given [S](using ToSeq[S, Int]): FromSeq[S, Int](v => new IntPointer(IntBuffer.wrap(v.toArray))) with ToInt32 with {}
  given [S](using ToSeq[S, Long]): FromSeq[S, Long](v => new LongPointer(LongBuffer.wrap(v.toArray))) with ToInt64 with {}
  given [S](using ToSeq[S, Float]): FromSeq[S, Float](v => new FloatPointer(FloatBuffer.wrap(v.toArray))) with ToFloat32 with {}
  given [S](using ToSeq[S, Double]): FromSeq[S, Double](v => new DoublePointer(DoubleBuffer.wrap(v.toArray))) with ToFloat64 with {}
  given [S](using ToSeq[S, Boolean]): FromSeq[S, Boolean](value => {
    val p = new BoolPointer(value.length)
    for (idx <- 0.until(value.length)) {
      p.put(idx, value(idx))
    }
    p
   }) with ToBool with {}

  abstract class FromSeq2D[S1, S2, V](using toSeq1: ToSeq[S1, S2], toSeq2: ToSeq[S2, V], fromScala: FromScala[Seq[V]]) extends FromScala[S1] {
    type OutputShape = ToShape[S1]

    override def apply[T <: DType](value: S1, dtype: T): Tensor[OutputShape, T] = {
      val seqs1 = toSeq1(value)
      val seq = seqs1.map(s => toSeq2(s)).flatten
      new Tensor(fromScala(seq, dtype).native.view(seqs1.size, seq.length / seqs1.size))
    }
  }

  given [S1, S2](using ToSeq[S1, S2], ToSeq[S2, Byte]): FromSeq2D[S1, S2, Byte] with ToInt8 with {}
  given [S1, S2](using ToSeq[S1, S2], ToSeq[S2, Short]): FromSeq2D[S1, S2, Short] with ToInt16 with {}
  given [S1, S2](using ToSeq[S1, S2], ToSeq[S2, Int]): FromSeq2D[S1, S2, Int] with ToInt32 with {}
  given [S1, S2](using ToSeq[S1, S2], ToSeq[S2, Long]): FromSeq2D[S1, S2, Long] with ToInt64 with {}
  given [S1, S2](using ToSeq[S1, S2], ToSeq[S2, Float]): FromSeq2D[S1, S2, Float] with ToFloat32 with {}
  given [S1, S2](using ToSeq[S1, S2], ToSeq[S2, Double]): FromSeq2D[S1, S2, Double] with ToFloat64 with {}
  given [S1, S2](using ToSeq[S1, S2], ToSeq[S2, Boolean]): FromSeq2D[S1, S2, Boolean] with ToBool with {}

  // TODO rewrite this recursively against >3 dimensions
  abstract class FromSeq3D[S1, S2, S3, V](using toSeq1: ToSeq[S1, S2], toSeq2: ToSeq[S2, S3], toSeq3: ToSeq[S3, V], fromScala: FromScala[Seq[V]]) extends FromScala[S1] {
    type OutputShape = ToShape[S1]

    override def apply[T <: DType](value: S1, dtype: T): Tensor[OutputShape, T] = {
      val seq = toSeq1(value).map(s2 => toSeq2(s2).map(s3 => toSeq3(s3)))
      new Tensor(fromScala(seq.flatten.flatten, dtype).native.view(seq.size, seq.head.size, seq.head.head.size))
    }
  }

  given [S1, S2, S3](using ToSeq[S1, S2], ToSeq[S2, S3], ToSeq[S3, Byte]): FromSeq3D[S1, S2, S3,Byte] with ToInt8 with {}
  given [S1, S2, S3](using ToSeq[S1, S2], ToSeq[S2, S3], ToSeq[S3, Short]): FromSeq3D[S1, S2, S3, Short] with ToInt16 with {}
  given [S1, S2, S3](using ToSeq[S1, S2], ToSeq[S2, S3], ToSeq[S3, Int]): FromSeq3D[S1, S2, S3, Int] with ToInt32 with {}
  given [S1, S2, S3](using ToSeq[S1, S2], ToSeq[S2, S3], ToSeq[S3, Long]): FromSeq3D[S1, S2, S3, Long] with ToInt64 with {}
  given [S1, S2, S3](using ToSeq[S1, S2], ToSeq[S2, S3], ToSeq[S3, Float]): FromSeq3D[S1, S2, S3, Float] with ToFloat32 with {}
  given [S1, S2, S3](using ToSeq[S1, S2], ToSeq[S2, S3], ToSeq[S3, Double]): FromSeq3D[S1, S2, S3, Double] with ToFloat64 with {}
  given [S1, S2, S3](using ToSeq[S1, S2], ToSeq[S2, S3], ToSeq[S3, Boolean]): FromSeq3D[S1, S2, S3, Boolean] with ToBool with {}

}
