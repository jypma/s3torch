package net.ypmania.s3torch

import org.bytedeco.pytorch.global.torch

// This can't be an enum, since then "val t = Int8 is typed DType, not Int8.type"
/** The data type for a tensor, e.g. 32-bit float or 8-bit integer. */
sealed abstract class DType(private[s3torch] val native: torch.ScalarType) {
  DType.fromNative = DType.fromNative + (native.value -> this)
}

object DType {
  // We can't use the actual enum instance as key, since somehow the libtorch wrapper creates new
  // enum instances that don't equal the constants we use below.
  private var fromNative = Map.empty[Byte, DType]

  case object BFloat16 extends DType(torch.ScalarType.BFloat16)
  case object Bool extends DType(torch.ScalarType.Bool)
  case object Int8 extends DType(torch.ScalarType.Char)
  case object Int16 extends DType(torch.ScalarType.Short)
  case object Int32 extends DType(torch.ScalarType.Int)
  case object Int64 extends DType(torch.ScalarType.Long)
  case object Float16 extends DType(torch.ScalarType.Half)
  case object Float32 extends DType(torch.ScalarType.Float)
  case object Float64 extends DType(torch.ScalarType.Double)
  case object UInt8 extends DType(torch.ScalarType.Byte)
  case object Undefined extends DType(torch.ScalarType.Undefined)

  def of(native: torch.ScalarType): DType = {
    fromNative.getOrElse(native.value, Undefined)
  }


  type Promoted[T <: DType, U <: DType] <: DType = (T, U) match {
    case (T, T)                                    => T
    case (U, U)                                    => U
    case (Undefined.type, U) | (T, Undefined.type)           => Undefined.type
    case (Bool.type, U)                                 => U
    case (T, Bool.type)                                 => T
    case (Int8.type, UInt8.type) | (UInt8.type, Int8.type)             => Int16.type
    case (UInt8.type, U)                                => U
    case (T, UInt8.type)                                => T
    case (Int8.type, U)                                 => U
    case (T, Int8.type)                                 => T
    case (Int16.type, U)                                => U
    case (T, Int16.type)                                => T
    case (Int32.type, U)                                => U
    case (T, Int32.type)                                => T
    case (Int64.type, U)                                => U
    case (T, Int64.type)                                => T
    //case (Float8_e5m2, U)                          => U
    //case (T, Float8_e5m2)                          => T
    //case (Float8_e4m3fn, U)                        => U
    //case (T, Float8_e5m2)                          => T
    case (Float16.type, BFloat16.type) | (BFloat16.type, Float16.type) => Float32.type
    case (Float16.type, U)                              => U
    case (T, Float16.type)                              => T
    case (Float32.type, U)                              => U
    case (T, Float32.type)                              => T
    case (Float64.type, U)                              => U
    case (T, Float64.type)                              => T
    // case (Complex32, U)                            => U
    // case (T, Complex32)                            => T
    // case (Complex64, U)                            => U
    // case (T, Complex64)                            => T
    // case (Complex128, U)                           => U
    // case (T, Complex128)                           => T
    case _                                         => DType
  }

  /** A floating-point or complex type (which is required for operations
    * like mean, std and others, and which is required where gradients
    * are calculated.) */
  type Floaty = BFloat16.type | Float16.type | Float32.type | Float64.type
}
