package net.ypmania.s3torch

import net.ypmania.s3torch.DType.Float32
import net.ypmania.s3torch.DType.Int32
import net.ypmania.s3torch.DType.Bool

case class Default[+T](value: T) {

}

trait DefaultInactiveGivens {
  given cuda: Default[Device.CUDA.type] = Default(Device.CUDA)
  given int32: Default[Int32] = Default(DType.int32)
  given bool: Default[Bool] = Default(DType.bool)
}

object Default extends DefaultInactiveGivens {
  /** Fallback default for DType. Define a given at local scope to override this. */
  given float32: Default[Float32] = Default(DType.float32)

  /** Fallback default for Device. Define a given at local scope to override this. */
  given cpu: Default[Device.CPU.type] = Default(Device.CPU)
}
