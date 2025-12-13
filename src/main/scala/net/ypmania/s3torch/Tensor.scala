package net.ypmania.s3torch

import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch

import org.bytedeco.javacpp.DoublePointer
import java.nio.DoubleBuffer

import internal.ZerosApply
import internal.FromScala
import internal.ToScala
import internal.Flatten
import internal.Broadcast
import internal.TensorOperand

import scala.collection.immutable.ArraySeq

import Shape.Scalar
import net.ypmania.s3torch.internal.FromScala.ToScalar
import net.ypmania.s3torch.internal.Torch
import net.ypmania.s3torch.Dim.DimArg

class Tensor[S <: Tuple, T <: DType](val native: pytorch.Tensor) {
  type Shape = S
  type DType = T

  import Tensor.*
  import Tuple.:*

  def flatten: Tensor[Flatten.All[S], T] = new Tensor[Flatten.All[S], T](native.flatten())

  def floor: Tensor[S, T] = new Tensor(native.floor())
  def floor_divide[V](value: V)(using op: TensorOperand[V]): op.Out[S, T] = op(this, value, _.floor_divide(_), _.floor_divide(_))
  def remainder[V](value: V)(using op: TensorOperand[V]): op.Out[S, T] = op(this, value, _.remainder(_), _.remainder(_))
  def size: Seq[Long] = ArraySeq.unsafeWrapArray(native.sizes.vec.get)

  def unsqueezeAfter[D <: Dim](d: D)(using ValueOf[Shape.IndexOf[S, D]]): Tensor[Shape.InsertAfter[S, Dim.One, D], T] =
    new Tensor(native.unsqueeze(valueOf[Shape.IndexOf[S, D]]))

  def unsqueezeAfterLast(using v:ValueOf[Tuple.Size[S]]): Tensor[S :* Dim.One, T] = new Tensor(native.unsqueeze(v.value))

  def value(using toScala: ToScala[S, T]) = toScala(native)


  def maxBy[D](using rm: RemoveDim[S, D]): Tensor[rm.OutputShape, T] = {
    ???
  }

  def maxBy[D](dim: D)(using rm: RemoveDim[S, D]): Tensor[rm.OutputShape, T] = {
    ???
  }

  def +[V](value: V)(using op: TensorOperand[V]): op.Out[S,T] = op(this, value, _.add(_), _.add(_))
  def -[V](value: V)(using op: TensorOperand[V]): op.Out[S,T] = op(this, value, _.sub(_), _.sub(_))
  def *[V](value: V)(using op: TensorOperand[V]): op.Out[S,T] = op(this, value, _.mul(_), _.mul(_))
  def /[V](value: V)(using op: TensorOperand[V]): op.Out[S,T] = op(this, value, _.div(_), _.div(_))

  private[Tensor] def unsafeWithShape[S1 <: Tuple]: Tensor[S1, T] = this.asInstanceOf[Tensor[S1, T]]
}

/** Math functions like sin, exp, are definied here, since "sin(x)"
  * approximated mathemetical notation better than "x.sin", even
  * though the latter would be more idiomatic Scala. */
object Tensor {
  def apply[V](value: V)(using fromScala: FromScala[V]): Tensor[fromScala.OutputShape, fromScala.DefaultDType] =
    fromScala(value, fromScala.defaultDType)
  def apply[V, T <: DType](value: V, dtype: T)(using fromScala: FromScala[V]): Tensor[fromScala.OutputShape, T] =
    fromScala(value, dtype)

  def arangeOf[D <: Dim](dim: D)(using d: DimArg[D]): Tensor[Tuple1[d.Out], Int64] = arange(0L, dim.size, 1L).unsafeWithShape
  def arangeOf[D <: Dim, T <: DType](dim: D, dtype: T)(using a: DimArg[D]): Tensor[Tuple1[a.Out], T] = arange(0L, dim.size, 1L, dtype).unsafeWithShape

  def arange[V](start: V, end: V, step: V)(using toScalar: ToScalar[V], fromScala: FromScala[V]): Tensor[Tuple1[Dim.Dynamic], fromScala.DefaultDType] = {
    new Tensor(torch.torch_arange(toScalar(start), toScalar(end), toScalar(step), Torch.tensorOptions(fromScala.defaultDType)))
  }
  def arange[V, T <: DType](start: V, end: V, step: V, dtype: T)(using toScalar: ToScalar[V]): Tensor[Tuple1[Dim.Dynamic], T] = {
    new Tensor(torch.torch_arange(toScalar(start), toScalar(end), toScalar(step), Torch.tensorOptions(dtype)))
  }

  def cos[S <: Tuple, T <: DType](t: Tensor[S, T]): Tensor[S, T] = new Tensor(t.native.cos)
  def exp[S <: Tuple, T <: DType](t: Tensor[S, T]): Tensor[S, T] = new Tensor(t.native.exp)
  def sin[S <: Tuple, T <: DType](t: Tensor[S, T]): Tensor[S, T] = new Tensor(t.native.sin)

  def zeros[T <: DType](using dtype: DefaultV2.DType[T]) = new ZerosApply(dtype.value)

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

  // Operations that only exist on scalars
  extension[T <: DType](t: Tensor[Scalar, T]) {
    def backward(): Unit = t.native.backward()
  }

  extension [S <: Tuple, T <: DType](t: Tensor[S, T])(using sq:Squeeze[S]) {
    def squeeze: Tensor[sq.OutputShape, T] = {
      ???
    }
  }

  // Target one dimension and remove it (used by max[dim=N, keepdim=false], squeeze[dims=...], etc.)

  trait RemoveDim[-S <: Tuple, D] {
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
}
