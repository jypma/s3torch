package net.ypmania.s3torch.internal

import java.nio.ByteBuffer
import java.nio.ShortBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.FloatBuffer
import java.nio.DoubleBuffer

import net.ypmania.s3torch.*
import net.ypmania.s3torch.Tensor.*
import net.ypmania.s3torch.Shape.*

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

import net.ypmania.s3torch.DType.*

trait FromScala[V] {
  type OutputShape <: Tuple
  type DefaultDType <: DType
  def apply[T <: DType](value: V): Tensor[OutputShape, T]
  def defaultDType: DefaultDType
}

object FromScala {
  // We need to explicitly extend these traits directly for each given. If we pull them in using givens, the types don't resolve.
  trait ToBool {
    type DefaultDType = Bool.type
    def defaultDType = Bool
  }

  trait ToInt8 {
    type DefaultDType = Int8.type
    def defaultDType = Int8
  }

  trait ToInt16 {
    type DefaultDType = Int16.type
    def defaultDType = Int16
  }

  trait ToInt32 {
    type DefaultDType = Int32.type
    def defaultDType = Int32
  }

  trait ToInt64 {
    type DefaultDType = Int64.type
    def defaultDType = Int64
  }

  trait ToFloat32 {
    type DefaultDType = Float32.type
    def defaultDType = Float32
  }

  trait ToFloat64 {
    type DefaultDType = Float64.type
    def defaultDType = Float64
  }

  trait ToScalar[V] extends (V => pytorch.Scalar)
  given ToScalar[Byte] with { def apply(v:Byte) = pytorch.Scalar(v) }
  given ToScalar[Short] with { def apply(v:Short) = pytorch.Scalar(v) }
  given ToScalar[Int] with { def apply(v:Int) = pytorch.Scalar(v) }
  given ToScalar[Long] with { def apply(v:Long) = pytorch.Scalar(v) }
  given ToScalar[Float] with { def apply(v:Float) = pytorch.Scalar(v) }
  given ToScalar[Double] with { def apply(v:Double) = pytorch.Scalar(v) }
  given ToScalar[Boolean] with { def apply(v:Boolean) = pytorch.AbstractTensor.create(v).item() }

  abstract class FromPrimitive[V](using toScalar: ToScalar[V]) extends FromScala[V] {
    type OutputShape = Scalar

    override def apply[T <: DType](value: V): Tensor[Scalar, T] = {
      val tensor = torch.scalar_tensor(
        toScalar(value),
        Torch.tensorOptions(defaultDType)
      )
      new Tensor(tensor)
    }
  }

  given FromPrimitive[Byte] with ToInt8 with {}
  given FromPrimitive[Short] with ToInt16 with {}
  given FromPrimitive[Int] with ToInt32 with {}
  given FromPrimitive[Long] with ToInt64 with {}
  given FromPrimitive[Float] with ToFloat32 with {}
  given FromPrimitive[Double] with ToFloat64 with {}
  given FromPrimitive[Boolean] with ToBool with {}

  type ToShape[V] <: Tuple = V match {
    case Seq[elem] => Dim.Dynamic *: ToShape[elem]
    case Tuple => Dim.Static[ToLong[Tuple.Size[V]]] *: ToShape[Tuple.Union[V]]
    case _ => EmptyTuple
  }

  abstract class FromSeq[S, V](toPointer: Seq[V] => Pointer)(using toSeq: ToSeq[S, V]) extends FromScala[S] {
    type OutputShape = ToShape[S]

    override def apply[T <: DType](value: S): Tensor[OutputShape, T] = {
      val seq = toSeq(value)
      val tensor = torch
        .from_blob(toPointer(seq), Array(seq.length.toLong), Torch.tensorOptions(defaultDType))
        .clone() // from_blob, if running on CPU, retains a reference to the original ByteBuffer, which might be GC'ed.
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

    override def apply[T <: DType](value: S1): Tensor[OutputShape, T] = {
      val seqs1 = toSeq1(value)
      val seq = seqs1.map(s => toSeq2(s)).flatten
      new Tensor(fromScala(seq).native.view(seqs1.size, seq.length / seqs1.size))
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

    override def apply[T <: DType](value: S1): Tensor[OutputShape, T] = {
      val seq = toSeq1(value).map(s2 => toSeq2(s2).map(s3 => toSeq3(s3)))
      new Tensor(fromScala(seq.flatten.flatten).native.view(seq.size, seq.head.size, seq.head.head.size))
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
