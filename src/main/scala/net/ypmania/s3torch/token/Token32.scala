package net.ypmania.s3torch.token

import net.ypmania.s3torch
import net.ypmania.s3torch.DTypeOps
import net.ypmania.s3torch.internal.DTypeFor
import org.bytedeco.javacpp.IntPointer
import org.bytedeco.pytorch

import java.nio.IntBuffer

/** A wrapper for tokens that are represented as 32-bit integers */
case class Token32[T <: s3torch.DType](val value: Int) extends AnyVal

/** Trait that can be extended by a case object to create a type-safe token type. */
trait Token32Type { self =>
  /** The Scala type for this token type */
  type S = Token32[DType]

  def apply(value: Int): S = Token32(value)

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
    override def toPointer(v: Seq[S]) = {
      val buf = IntBuffer.wrap(v.map(_.value).toArray)
      (new IntPointer(buf), buf)
    }
    override def toScalar(t: pytorch.Tensor) = Token32(t.item_int)
    override def copyToArray(t: pytorch.Tensor, a: Array[S]) = {
      val buf = t.createBuffer[IntBuffer]
      var idx = 0
      while (idx < buf.limit()) {
        a(idx) = Token32(buf.get(idx))
        idx += 1
      }
    }

    override def toValue(double: Double) = Token32(double.toInt)
    override def toDouble(s: ScalaType) = s.value.toDouble
    override def toValue(long: Long) = Token32(long.toInt)
    override def toLong(s: ScalaType) = s.value.toLong
  }
}
