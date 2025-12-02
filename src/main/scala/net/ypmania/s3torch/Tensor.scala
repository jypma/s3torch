package net.ypmania.s3torch

import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch

import internal.NativeConverters
import org.bytedeco.javacpp.DoublePointer
import java.nio.DoubleBuffer

import internal.ZerosApply
import internal.FromNative

class Tensor[S <: Tuple, T <: DType](delegate: pytorch.Tensor) {
  import Tensor.*

  def maxBy[D](using rm: RemoveDim[S, D]): Tensor[rm.OutputShape, T] = {
    ???
  }

  def maxBy[D](dim: D)(using rm: RemoveDim[S, D]): Tensor[rm.OutputShape, T] = {
    ???
  }
}

/*
trait Dimension {
  def size: Long
}

object Dimension {
  case class Known[S <: Singleton & Long](size: S) extends Dimension
}

sealed trait Shape {
  def size: Seq[Long]
}
object Shape {
  type Scalar = ∅.type
  case object ∅ extends Shape {
    override def size = Seq.empty
  }
  case class Of[D <: Dimension, Next <: Shape](d: Dimension, next: Shape) extends Shape {
    override def size = d.size +: next.size
  }
}
import Shape.Scalar
import Shape.Of

// TODO consider an opaque type for Tensor
// TODO see if we can make DType singleton types
class Tensor[S <: Shape, T <: DType](delegate: pytorch.Tensor) {

}

object Tensor {
  def scalar(scalar: Double,
    layout: Layout = Layout.Strided,
    device: Device = Device.CPU,
    requiresGrad: Boolean = false
  ): Tensor[Shape.Scalar, Float64] = {
    type T = ScalaToDType[Double]
    val tensor = torch.scalar_tensor(
      NativeConverters.toScalar(scalar),
      NativeConverters.tensorOptions(deriveDType[T], layout, device, requiresGrad)
    )
    Tensor[Scalar, T](tensor)
  }

  /*
   // require function of Long => D, and make that its shape
   // or, acknowledge that we have an unknown dimension here.
  def d1[D <: Dimension](values: Seq[Double],
    layout: Layout = Layout.Strided,
    device: Device = Device.CPU,
    requiresGrad: Boolean = false
  ) = {
    type T = ScalaToDType[Double]
    val ptr = new DoublePointer(DoubleBuffer.wrap(values.toArray))
    // TODO figure out why torch creates on CPU and moves later, see https://github.com/bytedeco/storch/commit/548de54d21b55d240b5f8b6c356a830dd62b6052
    val tensor = torch.from_blob(
      ptr,
      Array(values.length.toLong),
      NativeConverters.tensorOptions(deriveDType[T], layout, device, requiresGrad)
    )
    // The Java array is directly referenced from the torch pointer, but will later be GC'ed.
    // So, we have to clone the buffer.
      .clone()
    Tensor[Of[D, ∅], T](tensor)
   }
   */

  def from1d[D <: Dimension](values: Seq[Double])(mkDim: Int => Dimension): Tensor[Of[D, Scalar], ScalaToDType[Double]] = {
    ???
  }

  // TODO find a way to pass dtype as given with overridable defaults, perhaps together with the other props.
  def zeros1d[D <: Dimension, T <: DType](
    dim: D,
    layout: Layout = Layout.Strided,
    device: Device = Device.CPU,
    requiresGrad: Boolean = false
  ): Tensor[Of[D, Scalar], T] = {
    ???
  }

   val t = scalar(5.0)
}
 */

object Tensor {
  // TODO track what comes out of https://contributors.scala-lang.org/t/syntax-for-type-tuple-with-one-element/6974
  // A dimension is either statically known, or unknown
  /*
  trait Dim {
    def size: Long
  }
  trait DimLowPriorityGivens {
    given fromLongDynamic[L <: Long]: Conversion[L, Dynamic] with {
      def apply(l: L) = Dynamic(l)
    }
  }
  object Dim extends DimLowPriorityGivens {
    // The "+ 0L" hack here is needed, since scala 3.7.4 otherwise will allow Long variables to match here, even though
    // their compile-time value is unknown.
    import  scala.compiletime.ops.long.*
    given fromLongStatic[L <: Long & Singleton](using ValueOf[L], ValueOf[L + 0L]): Conversion[L, Static[L]] with {
      def apply(l: L) = new Static[L] {
        override def size = valueOf[L]
      }
    }
   }
   */
  /*
  trait HasDim[T] {
    type D <: Dim
    def size: Long
  }
  object HasDim {
    given fromLong[S <: Long & Singleton](using ValueOf[S]): HasDim[S] with {
      type D = Static[S]
      def size = valueOf[S]
    }
  }
   */

  //case class Static[S <: Singleton & Long](size: S) extends Dim
  /*
  abstract class Static[S <: Singleton & Long](using ValueOf[S]) extends Dim {
    type Size = S
    def size = valueOf[S]
   }
   */
  //def staticDim[S <: Singleton & Long](using ValueOf[S]): Static[S] = new Static[S] {
  //  override def size = valueOf[S]
  //}
  //type DimOf1 = Static[1L]
  //case class Dynamic(size: Long) extends Dim

  type Scalar = EmptyTuple

  // The default here is explicitly the global default, not depending on V. It's assumed that consistent tensors is more important than
  // predictable types here... that might need to be verified, though.
  def apply[V, T <: DType](value: V)(using fromNative: FromNative[V], t: DefaultV2.DType[T]): Tensor[fromNative.OutputShape, T] =
    fromNative.apply(value, t.value)

  def zeros[T <: DType](using dtype: Default[T]) = new ZerosApply[T]

  trait Squeeze[S <: Tuple] {
    type OutputShape <: Tuple
  }

  given [D1 <: Dim.One]: Squeeze[Tuple1[D1]] with {
    type OutputShape = Scalar
  }

  given squeezeD1ab[D1 <: Dim.One, D2 <: Dim.One]: Squeeze[(D1, D2)] with {
    type OutputShape = Scalar
  }

  given squeezeD2a[D1 <: Dim.One, D2 <: Dim]: Squeeze[(D1, D2)] with {
    type OutputShape = Tuple1[D2]
  }

  given squeezeD2b[D1 <: Dim, D2 <: Dim.One]: Squeeze[(D1, D2)] with {
    type OutputShape = Tuple1[D1]
  }

  extension [S <: Tuple, T <: DType](t: Tensor[S, T])(using sq:Squeeze[S]) {
    def squeeze: Tensor[sq.OutputShape, T] = {
      ???
    }
  }

  // Get the item of a scalar
  extension [T <: DType](t: Tensor[Scalar, T]) {
    def item: DTypeToScala[T] = {
      ???
    }
  }


  // Target one dimension and remove it (used by max[dim=N, keepdim=false], squeeze[dims=...], etc.)

  trait RemoveDim[S <: Tuple, D] {
    type OutputShape <: Tuple
  }

  given removeD1i[D1 <: Dim]: RemoveDim[Tuple1[D1], 0] with {
    type OutputShape = Scalar
  }

  given removeD1n[D1 <: Dim]: RemoveDim[Tuple1[D1], D1] with {
    type OutputShape = Scalar
  }

  given removeD2i0[D1 <: Dim, D2 <: Dim]: RemoveDim[(D1, D2), 0] with {
    type OutputShape = Tuple1[D2]
  }

  given removeD2n0[D1 <: Dim, D2 <: Dim]: RemoveDim[(D1, D2), D1] with {
    type OutputShape = Tuple1[D2]
  }

  given removeD2i1[D1 <: Dim, D2 <: Dim]: RemoveDim[(D1, D2), 1] with {
    type OutputShape = Tuple1[D1]
  }

  given removeD2n1[D1 <: Dim, D2 <: Dim]: RemoveDim[(D1, D2), D2] with {
    type OutputShape = Tuple1[D1]
  }

  // Test code (unnamed dimensions) --------------------------
  val scalar = Tensor(5.0)

  //val staticZeros1d = zeros.of[10L]
  //val staticZeros1dA = zeros.of[Static[10L]]
  //val staticZeros2d = zeros.of[(Static[10L], Static[5L])]

  //val mixedZeros1ds = zeros(staticDim[10L])

  val mixedZeros1ds1 = zeros(11L)
  val dynamicSize = 10L
  val mixedZeros1dd = zeros(dynamicSize)
  val mixedZeros2d = zeros(10L, Dim.Dynamic(10L))

  val of1 = zeros(1L)
  val of1Sq: Double = of1.squeeze.item // scalar
  val of1Max: Double = of1.maxBy[0].item // scalar

  val of1x1 = zeros(1L, 1L)
  val of1x1Sq: Double = of1x1.squeeze.item

  val of1x10 = zeros(1L, 10L)
  val of1x10Sq = of1x10.squeeze

  val of10x1 = zeros(10L, 1L)
  val of10x1sq = of10x1.squeeze

  val of10x10 = zeros(10L, 10L)
  // of10x10.squeeze // compile error

  // Test code (named dimensions)
  case object BatchSize extends Dim.Static[10L]

  //val of1xB = zeros.of[(Static[1L], BatchSize)]
  //val new1x1 = of1xB.maxBy[BatchSize]

  val of1xBa = zeros(1L, BatchSize)
  val new1a = of1xBa.maxBy[BatchSize.type]
  val new1b = of1xBa.maxBy(BatchSize)

  case class SomeUnkownDim(size: Long) extends Dim
  val of1xBb = zeros(1L, SomeUnkownDim(24))
  val new1c = of1xBb.maxBy[SomeUnkownDim]

}

/*
object Test extends App {
  trait Base {
    def value: Long
  }
  import  scala.compiletime.ops.long.*
  abstract class Known[L <: Long & Singleton](using ValueOf[L], ValueOf[L + 0L]) extends Base {
    override def value = valueOf[L]
    //def +[B <: Long & Singleton](that: Known[B])(using ValueOf[L + B]): Known[L + B] = new Known[L + B] {}
  }
  case class Unknown(value: Long) extends Base

  trait BaseLowPrio {
    given fromLongUnknown[L <: Long]: Conversion[L, Unknown] with {
      def apply(l: L) = Unknown(l)
    }
  }

  object Base extends BaseLowPrio {
    // The "+ 0L" hack here is needed, since scala 3.7.4 otherwise will allow Long variables to match here, even though
    // their compile-time value is unknown.
    given fromLong[L <: Long & Singleton](using ValueOf[L], ValueOf[L + 0L]): Conversion[L, Known[L]] with {
      def apply(l: L) = new Known[L] {
        override def value = {
          println("l=" + l)
          println("v=" + valueOf[L])
          l
        }
      }
    }
  }

  //def create[L <: Long & Singleton](v: L)(using ValueOf[L], ValueOf[L + 0L]): Known[L] = new Known[L] {}
  case class Wrapper[T](value: T)

  def create[B <: Base](b: B): Wrapper[B] = Wrapper(b)

  val n1 = create(1L) // Known[1L]
  val n2 = create(2L) // Known[1L]
  // val nr = create(System.currentTimeMillis()) // Found: Long, Required: Long & Singleton
  val runtimeValue = System.currentTimeMillis()
  val n3 = create(runtimeValue) // Known[runtimeValue.type]
  //println(n2.value)

  //val p1 = n1 + n2 // Known[3L]
  //val p2 = n1 + n3 // compile error, perhaps can be exploited to prevent injection of runtime values
/*
  def useGiven(b: Base) = {
    println(b.value)
  }

  useGiven(6L)
  // useGiven(System.currentTimeMillis()) // Found: Long, Required: Base

  case class Wrapper[T](value: T)

  def createWrapper[B <: Base](b: B): Wrapper[B] = Wrapper(b)

  val a = createWrapper(6L) // Wrapper[Known[6L]]
 //  val b = createWrapper(System.currentTimeMillis()) // Found: Long, Required: Base
 */
}
 */
