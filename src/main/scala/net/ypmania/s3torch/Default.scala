package net.ypmania.s3torch

import net.ypmania.s3torch.DType.Float32

case class Default[+T](value: T) {

}

object Default {
  /** Fallback default for DType. Define a given at local scope to override this. */
  given float32: Default[Float32.type] = Default(Float32)

  /** Fallback default for Device. Define a given at local scope to override this. */
  given cpu: Default[Device.CPU.type] = Default(Device.CPU)
}
