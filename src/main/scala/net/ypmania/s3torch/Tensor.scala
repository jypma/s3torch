package net.ypmania.s3torch

import net.ypmania.s3torch.Dim.{ |/, / }
import net.ypmania.s3torch.Shape.{SameSize, Elem}
import net.ypmania.s3torch.internal.FromScala.ToScalar
import net.ypmania.s3torch.internal.Torch
import net.ypmania.s3torch.internal.Untyped
import org.bytedeco.pytorch
import org.bytedeco.pytorch.ScalarTypeOptional
import org.bytedeco.pytorch.global.torch

import scala.collection.immutable.ArraySeq

import internal._
import Shape.Scalar
import DType._
import Device.CPU

class Tensor[S <: Tuple, T <: DType, D <: Device](val native: pytorch.Tensor) {
  type shape = S
  type dType = T
  type IdFn[T] = T => T
  /** A differently-shaped tensor with the same DType and Device */
  type Shaped[S1 <: Tuple] = Tensor[S1, T, D]
  /** A differently-shaped tensor, with different DType, on the same Device */
  type ShapedT[S1 <: Tuple, T1 <: DType] = Tensor[S1, T1, D]
  type This = Tensor[S, T, D]


  def ~>[U](f: This => U) = f(this)

  def dtype: T = DType.of(native.dtype().toScalarType()).asInstanceOf[T]

  def deviceType: DeviceType = DeviceType.of(native.device().`type`())

  def device: Device = new Device(deviceType, native.device().index()) {}

  private type BoolOp[V] = TensorOperandBool[S, T, D, V]
  /** Computes element-wise equality. We don't define pytorch's "eq" or "==", since those have a different meaning in Scala. */
  def #==[V](value: V)(using op:BoolOp[V]): op.Out = op(this, value, _.eq(_), _.eq(_))
  /** Computes element-wise nonequality. We don't define pytorch's "eq" or "!=" since those have a different meaning in Scala. */
  def #!=[V](value: V)(using op:BoolOp[V]): op.Out = op(this, value, _.ne(_), _.ne(_))
  /** Computes element-wise greater than. */
  def >[V](value: V)(using op:BoolOp[V]): op.Out = op(this, value, _.greater(_), _.greater(_))
  /** Computes element-wise less than. */
  def <[V](value: V)(using op:BoolOp[V]): op.Out = op(this, value, _.less(_), _.less(_))
  /** Computes element-wise greater than or equal. */
  def >=[V](value: V)(using op:BoolOp[V]): op.Out = op(this, value, _.greater_equal(_), _.greater_equal(_))
  /** Computes element-wise less than or equal. */
  def <=[V](value: V)(using op:BoolOp[V]): op.Out = op(this, value, _.less_equal(_), _.less_equal(_))

  /** True if `other` has the same size and elements as this tensor, false otherwise. */
  def equal[S2 <: Tuple](that: Shaped[S2])(using SameSize[S, S2]): Boolean = native.equal(that.native)

  override def equals(that: Any): Boolean = that match {
    // TODO investigate if this should just be .equal
    case other: Tensor[?, ?, ?] if dtype == other.dtype =>
      native.equal(other.native)
    case _ =>
      false
  }

  def flatten: Shaped[Flatten.All[S]] = new Tensor(native.flatten())

  def floor: This = new Tensor(native.floor())

  val log_softmax = new DimOperator.Of1Tensor[S, T, D] {
    type Out[Idx <: Int] = S
    def run[Idx <: Int](idx: Idx) = new Tensor(native.log_softmax(idx))
  }

  /** Fills elements of self tensor with value where mask is true. */
  def maskedFill[S2 <: Tuple, V](mask: ShapedT[S2, DType.Bool.type], value: V)(using Broadcast[S, S2, S])(using toScalar:FromScala.ToScalar[V]): Unit = {
    // Any [V] is indeed correct here, pytorch accepts doubles for int vectors.
    native.masked_fill_(mask.native, toScalar(value))
  }
  /** Returns copy that fills elements of self tensor with value where mask is true. */
  def maskedFilled[S2 <: Tuple, V, R <: Tuple](b: ShapedT[S2, DType.Bool.type], value: V)(using br:Broadcast[S, S2, R], toScalar:FromScala.ToScalar[V]): Shaped[R] = {
    new Tensor(native.masked_fill(b.native, toScalar(value)))
  }

  /** Matrix multiplication */
  // TODO double-check if this actually promotes the DType, or if it fails e.g. with Long times Float
  def matmul[S2 <: Tuple, T2 <: DType, R <: Tuple](b: ShapedT[S2, T2])(using MatMul[S, S2, R]): ShapedT[R, Promoted[T, T2]] =
    new Tensor(native.matmul(b.native))
  /** Matrix multiplication, alias for .matmul */
  def `@`[S2 <: Tuple, T2 <: DType, R <: Tuple](b: ShapedT[S2, T2])(using MatMul[S, S2, R]): ShapedT[R, Promoted[T, T2]] = matmul(b)

  def size: Seq[Long] = ArraySeq.unsafeWrapArray(native.sizes.vec.get)

  val softmax = new DimOperator.Of1Tensor[S, T, D] {
    type Out[Idx <: Int] = S
    def run[Idx <: Int](idx: Idx) = new Tensor(native.softmax(idx))
  }

  /** Transforms a split version of this tensor, split across dimension D in N parts. */
  val split = new DimOperator.Of1[S, T] {
    type Out[Idx <: Int] = SplitApply[Idx, Elem[S, Idx]]
    def run[Idx <: Int](idx: Idx) = new SplitApply(idx)
  }
  class SplitApply[Idx <: Int, D](idx: Idx) {
    /** Splits the selected dimension into N parts. */
    def into[N <: Long & Singleton](nn: N)(using dv: D |/ N, n: ValueOf[N]):
        Shaped[Shape.ReplaceWithTuple[S, (Dim.Static[N], Shape.Elem[S, Idx] / N), Idx]] = into

    /** Splits the selected dimension into N parts. */
    def into[N <: Long & Singleton](using dv: D |/ N, n: ValueOf[N]):
        Shaped[Shape.ReplaceWithTuple[S, (Dim.Static[N], Shape.Elem[S, Idx] / N), Idx]] = {
      val (before, after) = size.splitAt(idx)
      val dimsize = after.head
      val sizes = before :+ n.value :+ (dimsize / n.value) :++ after.tail
      new Tensor(native.view(sizes.toArray*))
    }
  }

  def to[D1 <: Device](device: D1): Tensor[S, T, D1] = new Tensor(native.to(device.native, dtype.native))

  def to[T1 <: DType](dtype: T1): Tensor[S, T1, D] = new Tensor(native.to(dtype.native))

  /** Swaps the given two dimensions. */
  val transpose = new DimOperator.Of2Tensor[S, T, D] {
    type Out[I1 <: Int, I2 <: Int] = Shape.Swap[S, I1, I2]
    def run[I1 <: Int, I2 <: Int](i1: I1, i2: I2) = new Tensor(native.transpose(i1, i2))
  }

  /** Swaps the last two dimensions. Tensor must have >= 2 dimensions. */
  def t[R <: Tuple](using Transpose[S, R]): Shaped[R] = {
    new Tensor(native.transpose(-2L, -1L))
  }

  /** Returns a view of this Tensor with just "Dim" as type for each
    * dimension. This makes it easier to create collections of
    * same-dimension but different length tensors. */
  def untyped(using ut: Untyped[S]): Shaped[ut.Out] = new Tensor(native)

  def update[I,V](indices: I, value: V)(using idx: Indices[S,I], updateSource: UpdateSource[V, D]): Unit = {
    updateSource(native, idx.toNative(indices), value)
  }

  /** Merges two dimensions that have previously been split off using split(). The selected dimension must be of type DividedDim, and must have a preceding
    * dimension with the remainder of the division. */
  val unsplit = new DimOperator.Of1Tensor[S, T, D] {
    type Out[Idx <: Int] = Unsplit[S, Idx]
    def run[Idx <: Int](idx: Idx) = {
      val (before, after) = size.splitAt(idx - 1)
      val sizes = before :+ (after(0) * after(1)) :++ after.drop(2)
      new Tensor(native.view(sizes.toArray*))
    }
  }

  /** Inserts a dimension of One after D */
  val unsqueezeAfter = new DimOperator.Of1Tensor[S, T, D] {
    type Out[Idx <: Int] = Shape.InsertAfter[S, Dim.One, Idx]
    def run[Idx <: Int](idx: Idx) = new Tensor(native.unsqueeze(idx + 1))
  }

  /** Inserts a dimension of One before D */
  val unsqueezeBefore = new DimOperator.Of1Tensor[S, T, D] {
    type Out[Idx <: Int] = Shape.InsertBefore[S, Dim.One, Idx]
    def run[Idx <: Int](idx: Idx) = new Tensor(native.unsqueeze(idx))
  }

  def value(using toScala: ToScala[S, T])(using D =:= CPU.type) = toScala(native)

  /** Applies [f] to [this] and the [opt] (if defined), or just returns [this] (if empty) */
  def when[A](opt: Option[A])(f: (This, A) => This): This = opt.map(a => f(this, a)).getOrElse(this)

  // --- Binary operands ----

  private type TensOp[V] = TensorOperand[S, T, D, V]
  private type ApplOp[V] = TensorOperandApply[S, T, D, V]
  /** Computes the division of this tensor with [value], elementwise, and takes floor() of the result. This is floor_divide in libtorch. */
  def /|/[V](value: V)(using op: TensOp[V]): op.Out = op(this, value, _.floor_divide(_), _.floor_divide(_))
  /** Computes the division of this tensor with [value], elementwise, takes floor() of the result, and reassigns to this tensor. This is floor_divide in libtorch. */
  def /|/=[V](value: V)(using op: ApplOp[V]): Unit = op(this, value, _.floor_divide_(_), _.floor_divide_(_))
  /** Calculates the remainder of division with the given value. */
  def %[V](value: V)(using op: TensOp[V]): op.Out = op(this, value, _.remainder(_), _.remainder(_))
  /** Calculates the remainder of division with the given value, and reassigns to this tensor. */
  def %=[V](value: V)(using op: ApplOp[V]): Unit = op(this, value, _.remainder_(_), _.remainder_(_))
  def +[V](value: V)(using op: TensOp[V]): op.Out = op(this, value, _.add(_), _.add(_))
  def +=[V](value: V)(using op: ApplOp[V]): Unit = op(this, value, _.add_(_), _.add_(_))
  def -[V](value: V)(using op: TensOp[V]): op.Out = op(this, value, _.sub(_), _.sub(_))
  def -=[V](value: V)(using op: ApplOp[V]): Unit = op(this, value, _.sub_(_), _.sub_(_))
  def *[V](value: V)(using op: TensOp[V]): op.Out = op(this, value, _.mul(_), _.mul(_))
  def *=[V](value: V)(using op: ApplOp[V]): Unit = op(this, value, _.mul_(_), _.mul_(_))
  def /[V](value: V)(using op: TensOp[V]): op.Out = op(this, value, _.div(_), _.div(_))
  def /=[V](value: V)(using op: ApplOp[V]): Unit = op(this, value, _.div_(_), _.div_(_))

  private[Tensor] def unsafeWithShape[S1 <: Tuple]: Shaped[S1] = this.asInstanceOf
}

/** Math functions like sin, exp, are definied here, since "sin(x)"
  * approximated mathemetical notation better than "x.sin", even
  * though the latter would be more idiomatic Scala. */
object Tensor {
  val KeepDim = ReduceOperand.KeepDim

  // TODO Revisit this, just always make the DType from a Default given.
  def apply[V, D <: Device](value: V)(using fromScala: FromScala[V], device: Default[D]): Tensor[fromScala.OutputShape, fromScala.DefaultDType, D] =
    fromScala(value, device.value)
  def apply[V, T <: DType, D <: Device](value: V, dtype: T)(using fromScala: FromScala[V], device: Default[D]): Tensor[fromScala.OutputShape, T, D] =
    fromScala(value, device.value).to(dtype)

  def arangeOf[D <: Dim, T <: DType, Dv <: Device](dim: D)(using dtype: Default[T], dv: Default[Dv]): Tensor[Tuple1[D], T, Dv] = arange(0L, dim.size, 1L, dtype.value).unsafeWithShape
  def arangeOf[D <: Dim, T <: DType, Dv <: Device](dim: D, dtype: T)(using Default[Dv]): Tensor[Tuple1[D], T, Dv] = arange(0L, dim.size, 1L, dtype).unsafeWithShape

  def arange[V, Dv <: Device](start: V, end: V, step: V)(using toScalar: ToScalar[V], fromScala: FromScala[V], dv: Default[Dv]): Tensor[Tuple1[Dim.Dynamic], fromScala.DefaultDType, Dv] = {
    new Tensor(torch.torch_arange(toScalar(start), toScalar(end), toScalar(step), Torch.tensorOptions(fromScala.defaultDType, dv.value)))
  }
  def arange[V, T <: DType, Dv <: Device](start: V, end: V, step: V, dtype: T)(using toScalar: ToScalar[V], dv:Default[Dv]): Tensor[Tuple1[Dim.Dynamic], T, Dv] = {
    new Tensor(torch.torch_arange(toScalar(start), toScalar(end), toScalar(step), Torch.tensorOptions(dtype, dv.value)))
  }

  // TODO consider a FunctionApply abstraction, to clean up duplication here
  def cos[S <: Tuple, T <: DType, D <: Device](t: Tensor[S, T, D]): Tensor[S, T, D] = new Tensor(t.native.cos)
  def exp[S <: Tuple, T <: DType, D <: Device](t: Tensor[S, T, D]): Tensor[S, T, D] = new Tensor(t.native.exp)
  def relu[S <: Tuple, T <: DType, D <: Device](t: Tensor[S, T, D]): Tensor[S, T, D] = new Tensor(t.native.relu)
  def sin[S <: Tuple, T <: DType, D <: Device](t: Tensor[S, T, D]): Tensor[S, T, D] = new Tensor(t.native.sin)

  def full[T <: DType, D <: Device, V](value: V)(using dtype: Default[T], device: Default[D], toScalar: ToScalar[V]) =
    new ZerosApply(dtype.value, device.value, torch.full(_, toScalar(value), _))
  def ones[T <: DType, D <: Device](using dtype: Default[T], device: Default[D]) =
    new ZerosApply(dtype.value, device.value, torch.torch_ones(_, _))
  def rand[T <: DType, D <: Device](using dtype: Default[T], device: Default[D], rnd:RandomSource) =
    rnd(new ZerosApply(dtype.value, device.value, torch.torch_rand(_, _)))
  def zeros[T <: DType, D <: Device](using dtype: Default[T], device: Default[D]) =
    new ZerosApply(dtype.value, device.value, torch.torch_zeros(_, _))

  // ---- Methods on Tensor that require floats
  extension[S <: Shape, T <: DType.Floaty, Dv <: Device](t: Tensor[S, T, Dv]) {
    // TODO consider a  ReduceOperandApply abstraction, in 0 and 1 arity, to clean up duplication here
    def stdBy[D, Idx <: Int, K <: ReduceOperand.Variant](dim: D, correction: Double = 1.0)(using keep: K)(using op: ReduceOperand[S,D,Idx,K]): t.Shaped[op.Out] =
      new Tensor(t.native.std(Array(op.index), new pytorch.ScalarOptional(new pytorch.Scalar(correction)), op.keep))
    def meanBy[D, Idx <: Int, K <: ReduceOperand.Variant](dim: D)(using keep: K)(using op: ReduceOperand[S,D,Idx,K]): t.Shaped[op.Out] =
      new Tensor(t.native.mean(Array(op.index), op.keep, new ScalarTypeOptional))
  }

  // ---- Methods on Tensor that only exist on scalars
  extension[T <: DType, D <: Device](t: Tensor[Scalar, T, D]) {
    def backward(): Unit = t.native.backward()
  }
}
