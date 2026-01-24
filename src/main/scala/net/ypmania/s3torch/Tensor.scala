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
import internal.UpdateSource
import internal.ReduceOperand
import internal.Unsplit
import internal.VerifyShape

import scala.collection.immutable.ArraySeq

import Shape.Scalar
import net.ypmania.s3torch.internal.FromScala.ToScalar
import net.ypmania.s3torch.internal.Torch
import net.ypmania.s3torch.Dim.*
import net.ypmania.s3torch.Shape.*
import DType.*
import org.bytedeco.pytorch.ScalarTypeOptional

class Tensor[S <: Tuple, T <: DType](val native: pytorch.Tensor) {
  type Shape = S
  type DType = T
  type IdFn[T] = T => T

  import Tensor.*
  import Tuple.:*

  def flatten: Tensor[Flatten.All[S], T] = new Tensor[Flatten.All[S], T](native.flatten())

  def floor: Tensor[S, T] = new Tensor(native.floor())

  def size: Seq[Long] = ArraySeq.unsafeWrapArray(native.sizes.vec.get)

  /** Transforms a split version of this tensor, split across dimension D in N parts, using the given function, while retaining the original
    type once computation is complete. */
  def split[D, Idx <: Int](d: D)(using sel: Shape.Select[S, D, Idx], idx: ValueOf[Idx]) = new SplitApply[Idx, Elem[S, Idx]](idx.value)
  class SplitApply[Idx <: Int, D](idx: Idx) {
    def apply[N <: Long & Singleton](using dv: D |/ N, n: ValueOf[N]):
        Tensor[Shape.ReplaceWithTuple[S, (Dim.Static[N], Shape.Elem[S, Idx] / N), Idx], T] = {
      val (before, after) = size.splitAt(idx)
      val dimsize = after.head
      val sizes = before :+ n.value :+ (dimsize / n.value) :++ after.tail
      new Tensor(native.view(sizes.toArray*))
    }
  }

  def to[T1 <: DType](dtype: T1): Tensor[S, T1] = new Tensor(native.to(dtype.scalarType))

  def transpose[D1, D2, Idx1 <: Int, Idx2 <: Int](d1: D1, d2: D2)(using s1: Shape.Select[S,D1,Idx1], i1: ValueOf[Idx1], s2:Shape.Select[S,D2,Idx2], i2: ValueOf[Idx2]): Tensor[Shape.Swap[S, Idx1, Idx2], T] = {
    new Tensor(native.transpose(i1.value, i2.value))
  }
  def transpose(using Is2D[S]): Tensor[Shape.Swap[S, 0, 1], T] = {
    // Somehow, defining this as an extension method on Tensor[(D1, D2)]fails to compile.
    new Tensor(native.transpose(0L, 1L))
  }
  def update[I,V](indices: I, value: V)(using idx: Indices[S,I], updateSource: UpdateSource[V]): this.type = {
    updateSource(native, idx.toNative(indices), value)
    this
  }

  /** Merges two dimensions that have previously been split off using split(). The selected dimension must be of type DividedDim, and must have a preceding
    * dimension with the remainder of the division. */
  def unsplit[D, Idx <: Int](d: D)(using sel: Shape.Select[S, D, Idx], idx: ValueOf[Idx])(using VerifyShape[Unsplit[S, Idx]]): Tensor[Unsplit[S, Idx], T] = {
    val (before, after) = size.splitAt(idx.value - 1)
    val sizes = before :+ (after(0) * after(1)) :++ after.drop(2)
    new Tensor(native.view(sizes.toArray*))
  }

  /** Inserts a dimension of One after D */
  def unsqueezeAfter[D, Idx <: Int](d: D)(using sel: Shape.Select[S,D,Idx], idx: ValueOf[Idx]): Tensor[Shape.InsertAfter[S, Dim.One, Idx], T] =
    new Tensor(native.unsqueeze(idx.value + 1))

  /** Inserts a dimension of One before D */
  def unsqueezeBefore[D, Idx <: Int](d: D)(using sel: Shape.Select[S,D,Idx], idx: ValueOf[Idx]): Tensor[Shape.InsertBefore[S, Dim.One, Idx], T] =
    new Tensor(native.unsqueeze(idx.value))

  def value(using toScala: ToScala[S, T]) = toScala(native)

  // --- Binary operands ----

  def floor_divide[V](value: V)(using op: TensorOperand[S, T, V]): op.Out = op(this, value, _.floor_divide(_), _.floor_divide(_))
  def remainder[V](value: V)(using op: TensorOperand[S, T, V]): op.Out = op(this, value, _.remainder(_), _.remainder(_))
  def +[V](value: V)(using op: TensorOperand[S, T, V]): op.Out = op(this, value, _.add(_), _.add(_))
  def -[V](value: V)(using op: TensorOperand[S, T, V]): op.Out = op(this, value, _.sub(_), _.sub(_))
  def *[V](value: V)(using op: TensorOperand[S, T, V]): op.Out = op(this, value, _.mul(_), _.mul(_))
  def /[V](value: V)(using op: TensorOperand[S, T, V]): op.Out = op(this, value, _.div(_), _.div(_))

  private[Tensor] def unsafeWithShape[S1 <: Tuple]: Tensor[S1, T] = this.asInstanceOf[Tensor[S1, T]]
}

/** Math functions like sin, exp, are definied here, since "sin(x)"
  * approximated mathemetical notation better than "x.sin", even
  * though the latter would be more idiomatic Scala. */
object Tensor {
  val KeepDim = ReduceOperand.KeepDim

  def apply[V](value: V)(using fromScala: FromScala[V]): Tensor[fromScala.OutputShape, fromScala.DefaultDType] =
    fromScala(value)
  def apply[V, T <: DType](value: V, dtype: T)(using fromScala: FromScala[V]): Tensor[fromScala.OutputShape, T] =
    fromScala(value).to(dtype)

  def arangeOf[D <: Dim](dim: D): Tensor[Tuple1[D], Int64.type] = arange(0L, dim.size, 1L).unsafeWithShape
  def arangeOf[D <: Dim, T <: DType](dim: D, dtype: T): Tensor[Tuple1[D], T] = arange(0L, dim.size, 1L, dtype).unsafeWithShape

  def arange[V](start: V, end: V, step: V)(using toScalar: ToScalar[V], fromScala: FromScala[V]): Tensor[Tuple1[Dim.Dynamic], fromScala.DefaultDType] = {
    new Tensor(torch.torch_arange(toScalar(start), toScalar(end), toScalar(step), Torch.tensorOptions(fromScala.defaultDType)))
  }
  def arange[V, T <: DType](start: V, end: V, step: V, dtype: T)(using toScalar: ToScalar[V]): Tensor[Tuple1[Dim.Dynamic], T] = {
    new Tensor(torch.torch_arange(toScalar(start), toScalar(end), toScalar(step), Torch.tensorOptions(dtype)))
  }

    // TODO consider a FunctionApply abstraction, to clean up duplication here
  def cos[S <: Tuple, T <: DType](t: Tensor[S, T]): Tensor[S, T] = new Tensor(t.native.cos)
  def exp[S <: Tuple, T <: DType](t: Tensor[S, T]): Tensor[S, T] = new Tensor(t.native.exp)
  def relu[S <: Tuple, T <: DType](t: Tensor[S, T]): Tensor[S, T] = new Tensor(t.native.relu)
  def sin[S <: Tuple, T <: DType](t: Tensor[S, T]): Tensor[S, T] = new Tensor(t.native.sin)

  def ones[T <: DType](using dtype: Default[T]) = new ZerosApply(dtype.value, torch.torch_ones(_, _))
  def zeros[T <: DType](using dtype: Default[T]) = new ZerosApply(dtype.value, torch.torch_zeros(_, _))
  def rand[T <: DType](using dtype: Default[T], rnd:RandomSource) = rnd(new ZerosApply(dtype.value, torch.torch_rand(_, _)))

  // ---- Methods on Tensor that require floats
  extension[S <: Shape, T <: DType.Floaty](t: Tensor[S, T]) {
    // TODO consider a  ReduceOperandApply abstraction, in 0 and 1 arity, to clean up duplication here
    def stdBy[D, Idx <: Int, K <: ReduceOperand.Variant](dim: D, correction: Double = 1.0)(using keep: K)(using op: ReduceOperand[S,D,Idx,K]): Tensor[op.Out, T] =
      new Tensor(t.native.std(Array(op.index), new pytorch.ScalarOptional(new pytorch.Scalar(correction)), op.keep))
    def meanBy[D, Idx <: Int, K <: ReduceOperand.Variant](dim: D)(using keep: K)(using op: ReduceOperand[S,D,Idx,K]): Tensor[op.Out, T] =
      new Tensor(t.native.mean(Array(op.index), op.keep, new ScalarTypeOptional))
  }

  // ---- Methods on Tensor that only exist on scalars
  extension[T <: DType](t: Tensor[Scalar, T]) {
    def backward(): Unit = t.native.backward()
  }

  // TODO refactor or remove
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
}
