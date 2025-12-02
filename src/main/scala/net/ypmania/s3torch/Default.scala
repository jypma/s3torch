package net.ypmania.s3torch

trait DefaultV2 {

}

trait DefaultLowPrioGivens {
  import DefaultV2.DType

  given float16: DType[Float16] = DType(net.ypmania.s3torch.float16)
}

object DefaultV2 extends DefaultLowPrioGivens {
  case class DType[D <: net.ypmania.s3torch.DType](value: D)

  given float32: DType[Float32] = DType(net.ypmania.s3torch.float32)
}
