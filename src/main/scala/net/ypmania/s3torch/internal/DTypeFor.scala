package net.ypmania.s3torch.internal

import net.ypmania.s3torch.DType
import net.ypmania.s3torch.DType.*

/** Given that gives the appropriate default DType for a Scala type (when used with Tensor.apply) */
trait DTypeFor[V] {
  type Out <: DType
  def dType: Out
}
object DTypeFor {
  given DTypeFor[Boolean] with {
    type Out = Bool
    def dType = DType.bool
  }
  given DTypeFor[Byte] with {
    type Out = Int8
    def dType = DType.int8
  }
  given DTypeFor[Short] with {
    type Out = Int16
    def dType = DType.int16
  }
  given int32: DTypeFor[Int] with {
    type Out = Int32
    def dType = DType.int32
  }
  given DTypeFor[Long] with {
    type Out = Int64
    def dType = DType.int64
  }
  given DTypeFor[Float] with {
    type Out = Float32
    def dType = DType.float32
  }
  given DTypeFor[Double] with {
    type Out = Float64
    def dType = DType.float64
  }
}
