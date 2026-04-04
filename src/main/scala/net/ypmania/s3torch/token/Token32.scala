package net.ypmania.s3torch.token

import net.ypmania.s3torch
import net.ypmania.s3torch.internal.DTypeFor
import net.ypmania.s3torch.DTypeOps

import org.bytedeco.pytorch
import org.bytedeco.javacpp.IntPointer
import java.nio.IntBuffer

/** A wrapper for tokens that are represented as 32-bit integers */
case class Token32[T <: s3torch.DType](val value: Int) extends AnyVal

/** Trait that can be extended by a case object to create a type-safe token type. */
trait Token32Type { self =>
  /** The Scala type for this token type */
  type S = Token32[DType]

  given Token[S] with {
    override def unknown = Token32(0)
    override def next(t: S) = Token32(t.value + 1)
    override def max(ts: Iterable[S]) = ts.maxBy(_.value)
  }

  /** The DType for this token type (as a type) */
  abstract class DType extends s3torch.DType.Int32
  /** The DType for this token type (as a value) */
  val dType = new DType {}

  given DTypeOps[DType, S] = new Token32.Ops[DType](dType)
  given DTypeFor[S] with {
    type Out = DType
    def dType = self.dType
  }
}

object Token32 {
  class Ops[T <: net.ypmania.s3torch.DType.Int32](val dType: T) extends DTypeOps[T, Token32[T]] {
    type S = Token32[T]
    override def fromScalar(v: S): pytorch.Scalar = pytorch.Scalar(v.value)
    override def toPointer(v: Seq[S]) = new IntPointer(IntBuffer.wrap(v.map(_.value).toArray))
    override def toScalar(t: pytorch.Tensor) = t.item_int.asInstanceOf[S]
    override def copyToArray(t: pytorch.Tensor, a: Array[S]) = t.createBuffer[IntBuffer].get(a.map(_.value))

    override def toValue(double: Double) = Token32(double.toInt).asInstanceOf[ScalaType]
    override def toDouble(s: ScalaType) = s.value.toDouble
    override def toValue(long: Long) = long.toByte.asInstanceOf[ScalaType]
    override def toLong(s: ScalaType) = s.value.toLong
  }
}
