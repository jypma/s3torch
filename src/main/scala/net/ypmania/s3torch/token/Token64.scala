package net.ypmania.s3torch.token

import net.ypmania.s3torch
import net.ypmania.s3torch.DTypeOps
import net.ypmania.s3torch.internal.DTypeFor
import org.bytedeco.pytorch

import org.bytedeco.javacpp.LongPointer
import java.nio.LongBuffer

/** A wrapper for tokens that are represented as 64-bit integers */
case class Token64[T <: s3torch.DType](val value: Long) extends AnyVal

/** Trait that can be extended by a case object to create a type-safe token type. */
trait Token64Type { self =>
  /** The Scala type for this token type */
  type S = Token64[DType]

  def apply(value: Long): S = Token64(value)

  given Token[S] with {
    override def unknown = Token64(0)
    override def next(t: S) = Token64(t.value + 1)
    override def max(ts: Iterable[S]) = ts.maxBy(_.value)
  }

  /** The DType for this token type (as a type) */
  abstract class DType extends s3torch.DType.Int64
  /** The DType for this token type (as a value) */
  val dType = new DType {}

  given DTypeOps[DType, S] = new Token64.Ops[DType](dType)
  given DTypeFor[S] with {
    type Out = DType
    def dType = self.dType
  }
}

object Token64 {
  class Ops[T <: net.ypmania.s3torch.DType.Int64](val dType: T) extends DTypeOps[T, Token64[T]] {
    type S = Token64[T]
    override def fromScalar(v: S): pytorch.Scalar = pytorch.Scalar(v.value)
    override def toPointer(v: Seq[S]) = new LongPointer(LongBuffer.wrap(v.map(_.value).toArray))
    override def toScalar(t: pytorch.Tensor) = Token64(t.item_long)
    override def copyToArray(t: pytorch.Tensor, a: Array[S]) = {
      val buf = t.createBuffer[LongBuffer]
      var idx = 0
      while (idx < buf.limit()) {
        a(idx) = Token64(buf.get(idx))
        idx += 1
      }
    }

    override def toValue(double: Double) = Token64(double.toLong)
    override def toDouble(s: ScalaType) = s.value.toDouble
    override def toValue(long: Long) = Token64(long)
    override def toLong(s: ScalaType) = s.value
  }
}
