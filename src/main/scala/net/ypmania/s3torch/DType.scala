package net.ypmania.s3torch

import org.bytedeco.pytorch.global.torch

// This can't be an enum, since then "val t = Int8 is typed DType, not Int8.type"
// It also isn't modelled as a sealed type, since we want users to be able to
// safely create named sub-types, e.g.:
// case object InputToken extends Int32
// to have a vector of "input tokens", which behaves like any int32 vector would,
// but can't be confused with a vector of e.g. output tokens.
/** The data type for a tensor, e.g. 32-bit float or 8-bit integer. */
abstract class DType(private[s3torch] val native: torch.ScalarType) {
  DType.fromNative = DType.fromNative + (native.value -> this)
}

object DType {
  // We can't use the actual enum instance as key, since somehow the libtorch wrapper creates new
  // enum instances that don't equal the constants we use below.
  private var fromNative = Map.empty[Byte, DType]

  abstract class BFloat16 extends DType(torch.ScalarType.BFloat16)
  val bfloat16 = new BFloat16 {}
  abstract class Bool extends DType(torch.ScalarType.Bool)
  val bool = new Bool {}
  abstract class Int8 extends DType(torch.ScalarType.Char)
  val int8 = new Int8 {}
  abstract class Int16 extends DType(torch.ScalarType.Short)
  val int16 = new Int16 {}
  abstract class Int32 extends DType(torch.ScalarType.Int)
  val int32 = new Int32 {}
  abstract class Int64 extends DType(torch.ScalarType.Long)
  val int64 = new Int64 {}
  abstract class Float16 extends DType(torch.ScalarType.Half)
  val float16 = new Float16 {}
  abstract class Float32 extends DType(torch.ScalarType.Float)
  val float32 = new Float32 {}
  abstract class Float64 extends DType(torch.ScalarType.Double)
  val float64 = new Float64 {}
  abstract class UInt8 extends DType(torch.ScalarType.Byte)
  val uint8 = new UInt8 {}
  case object Undefined extends DType(torch.ScalarType.Undefined)

  def of(native: torch.ScalarType): DType = {
    fromNative.getOrElse(native.value, Undefined)
  }


  type Promoted[T <: DType, U <: DType] <: DType = (T, U) match {
    case (T, T)                                    => T
    case (U, U)                                    => U
    case (Undefined.type, U) | (T, Undefined.type)           => Undefined.type
    case (Bool, U)                                 => U
    case (T, Bool)                                 => T
    case (Int8, UInt8) | (UInt8, Int8)             => Int16
    case (UInt8, U)                                => U
    case (T, UInt8)                                => T
    case (Int8, U)                                 => U
    case (T, Int8)                                 => T
    case (Int16, U)                                => U
    case (T, Int16)                                => T
    case (Int32, U)                                => U
    case (T, Int32)                                => T
    case (Int64, U)                                => U
    case (T, Int64)                                => T
    //case (Float8_e5m2, U)                          => U
    //case (T, Float8_e5m2)                          => T
    //case (Float8_e4m3fn, U)                        => U
    //case (T, Float8_e5m2)                          => T
    case (Float16, BFloat16) | (BFloat16, Float16) => Float32
    case (Float16, U)                              => U
    case (T, Float16)                              => T
    case (Float32, U)                              => U
    case (T, Float32)                              => T
    case (Float64, U)                              => U
    case (T, Float64)                              => T
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
  type Floaty = BFloat16 | Float16 | Float32 | Float64
}
