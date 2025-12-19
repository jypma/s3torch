package net.ypmania.s3torch

import net.ypmania.s3torch.DType

case class Default[+T](value: T) {

}

trait DefaultLowPrioGivens {
  //import DefaultV2.DType

  /*
  given int8: DType = net.ypmania.s3torch.int8
  given uint8: DType = net.ypmania.s3torch.uint8
  given int16: DType[Int16] = DType(net.ypmania.s3torch.int16)
  given int32: DType[Int32] = DType(net.ypmania.s3torch.int32)
  given int64: DType[Int64] = DType(net.ypmania.s3torch.int64)
  given float16: DType[Float16] = DType(net.ypmania.s3torch.float16)
   given float64: DType[Float64] = DType(net.ypmania.s3torch.float64)
   */
}

object Default extends DefaultLowPrioGivens {
  //case class DType[D <: net.ypmania.s3torch.DType](value: D)

  //given float32: DType = net.ypmania.s3torch.float32
  given float32: Default[net.ypmania.s3torch.DType.Float32.type] = Default(net.ypmania.s3torch.DType.Float32)
}
