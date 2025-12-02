package net.ypmania.s3torch

trait DefaultV2 {

}

trait DefaultLowPrioGivens {
  import DefaultV2.DType

  given int8: DType[Int8] = DType(net.ypmania.s3torch.int8)
  given uint8: DType[UInt8] = DType(net.ypmania.s3torch.uint8)
  given int16: DType[Int16] = DType(net.ypmania.s3torch.int16)
  given int32: DType[Int32] = DType(net.ypmania.s3torch.int32)
  given int64: DType[Int64] = DType(net.ypmania.s3torch.int64)
  given float16: DType[Float16] = DType(net.ypmania.s3torch.float16)
  given float64: DType[Float64] = DType(net.ypmania.s3torch.float64)
}

object DefaultV2 extends DefaultLowPrioGivens {
  case class DType[D <: net.ypmania.s3torch.DType](value: D)

  given float32: DType[Float32] = DType(net.ypmania.s3torch.float32)
}
