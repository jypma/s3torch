package net.ypmania.s3torch

import net.ypmania.s3torch.Shape.Elem
import net.ypmania.s3torch.Shape.SameSize
import org.bytedeco.pytorch
import org.bytedeco.pytorch.ScalarTypeOptional
import org.bytedeco.pytorch.global.torch

import scala.compiletime.ops.int.-
import scala.compiletime.ops.int.>=
import scala.reflect.ClassTag
import scala.util.Using

import internal._
import Shape.Scalar
import DType._
import Device.CPU
import Tuple.++

/**
  * A tensor is a multidimensional structure of values, wrapping pytorch's tensor. A tensor has the following properties, all of
  * which are tracked compile time by this Scala library (in addition to at runtime by pytorch itself):
  *
  * - Shape [S] (a Tuple of sub-types of Dim), which represent the dimensions of this tensor
  * - Data type [T], which represents the storage type for the values of this tensor
  * - Device [D], which represents the physical device on which the tensor is stored (e.g. the main CPU and memory, or a graphics card)
  *
  * All methods on Tensor guarantee that the types accurately reflect the tensor, so that any incompatibilities always
  * result in compile-time errors, rather than runtime discoveries.
  */
class Tensor[S <: Tuple, T <: DType, D <: Device](val native: pytorch.Tensor) {
  type shape = S
  type dType = T
  type IdFn[T] = T => T
  /** A differently-shaped tensor with the same DType and Device */
  type Shaped[S1 <: Tuple] = Tensor[S1, T, D]
  /** A differently-shaped tensor, with different DType, on the same Device */
  type ShapedT[S1 <: Tuple, T1 <: DType] = Tensor[S1, T1, D]
  type This = Tensor[S, T, D]

  /** An operation that reduces the tensor across a given dimension. */
  class ReduceOp(op: ((Long, Boolean) => pytorch.Tensor)) extends DimOperator.Of1Tensor[S, T, D] {
    type Out[Idx <: Int] = Shape.Remove[S, Idx]
    protected def run[Idx <: Int](idx: Int) = new Tensor(op(idx, false))

    /** Variant of this operation that keeps the targeted dimension (reduced to one), rather than removing it. */
    case object keepDim extends DimOperator.Of1Tensor[S, T, D] {
      type Out[Idx <: Int] = Shape.Replace[S, Dim.One, Idx]
      protected def run[Idx <: Int](idx: Int) = new Tensor(op(idx, true))
    }
  }

  /** Experimental syntax: returns the result of applying the given function to this tensor. This allows us
    * to write nested function applications as arrows instead. This is particularly expressive when
    * writing several layers of neural network transformations calling into each other. */
  def ~>[U](f: This => U) = f(this)

  class CatApply[U <: Tuple](that: Shaped[U]) {
    type Out[Idx <: Int, O] = Shape.Replace[S, O, Idx]
    type Pick[Idx <: Int] = Cat.PickDynamic[Tuple.Elem[S, Idx], Tuple.Elem[U, Idx]]

    protected def cat(a: pytorch.Tensor, b: pytorch.Tensor, idx: Int) = torch.cat(new pytorch.TensorVector(native, that.native), idx)

    def apply[Dm](d: Dm)(using sel: Shape.SelectIdx[S,Dm], pick: Pick[sel.Idx])(using VerifyShape[Out[sel.Idx, pick.Out]], Cat[S, U, sel.Idx]): Shaped[Out[sel.Idx, pick.Out]] = new Tensor(cat(native, that.native, sel.idx))

    def apply[Dm](using sel: Shape.SelectIdx[S,Dm], pick: Pick[sel.Idx])(using VerifyShape[Out[sel.Idx, pick.Out]], Cat[S, U, sel.Idx]): Shaped[Out[sel.Idx, pick.Out]]  = new Tensor(cat(native, that.native, sel.idx))
  }

  /** Concatenates two tensors along a given dimension:
    * val result = t1.cat(t2)(alongDim)
    */
  def cat[U <: Tuple](that: Shaped[U]) = new CatApply[U](that)

  def contiguous: This = new Tensor(native.contiguous())

  def dtype: DType = DType.of(native.dtype().toScalarType())

  def deviceType: DeviceType = DeviceType.of(native.device().`type`())

  def device: Device = Device.of(native.device())

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
  /** Computes element-wise logical AND, interpreting both sides as a boolean. */
  def &&[V](value: V)(using op:BoolOp[V]): op.Out = op(this, value, _.__and__(_), _.__and__(_))
  /** Computes element-wise logical OR, interpreting both sides as a boolean. */
  def ||[V](value: V)(using op:BoolOp[V]): op.Out = op(this, value, _.__or__(_), _.__or__(_))

  /** True if `other` has the same size and elements as this tensor, false otherwise, and fails to compile if
    * the tensors are known to have a different size, or a different device. */
  def equal[S2 <: Tuple](that: Shaped[S2])(using SameSize[S, S2]): Boolean = native.equal(that.native)

  override def equals(that: Any): Boolean = that match {
    // Check dtype and device, since pytorch will just crash if they don't match.
    case other: Tensor[?, ?, ?] if dtype == other.dtype && device == other.device =>
      native.equal(other.native)
    case _ =>
      false
  }

  def flatten(using f:Flatten[S]): Shaped[Tuple1[f.Out]] = new Tensor(native.flatten())

  def floor: This = new Tensor(native.floor())

  val log_softmax = new DimOperator.Of1Tensor[S, T, D] {
    type Out[Idx <: Int] = S
    def run[Idx <: Int](idx: Int) = new Tensor(native.log_softmax(idx))
  }

  /** Fills elements of self tensor with value where mask is true. */
  def maskedFill[S2 <: Tuple, V](mask: ShapedT[S2, DType.Bool], value: V)(using Broadcast[S, S2, S])(using ops: DTypeOps.Scalar[T, V]): Unit = {
    // Any [V] is indeed correct here, pytorch accepts doubles for int vectors.
    native.masked_fill_(mask.native, ops.fromScalar(value))
  }
  /** Returns copy that fills elements of self tensor with value where mask is true. */
  def maskedFilled[S2 <: Tuple, V, R <: Tuple](b: ShapedT[S2, DType.Bool], value: V)(using br:Broadcast[S, S2, R], ops: DTypeOps.Scalar[T, V]): Shaped[R] = {
    new Tensor(native.masked_fill(b.native, ops.fromScalar(value)))
  }

  /** Matrix multiplication */
  // TODO double-check if this actually promotes the DType, or if it fails e.g. with Long times Float
  def matmul[S2 <: Tuple, T2 <: DType, R <: Tuple](b: ShapedT[S2, T2])(using MatMul[S, S2, R]): ShapedT[R, Promoted[T, T2]] =
    new Tensor(native.matmul(b.native))
  /** Matrix multiplication, alias for .matmul */
  def `@`[S2 <: Tuple, T2 <: DType, R <: Tuple](b: ShapedT[S2, T2])(using MatMul[S, S2, R]): ShapedT[R, Promoted[T, T2]] = matmul(b)

  /** Returns the maximum value of all elements. */
  def max: Shaped[Scalar] = new Tensor(native.max)

  case class MaxResult[S <: Tuple](result: Shaped[S], indices: Tensor[S, Int64, D])

  // Special variant of ReduceOp that returns a Tuple (MaxResult), rather than a single tensor.
  class MaxReduceOp extends DimOperator.Of1[S, T] {
    type Out[Idx <: Int] = MaxResult[Shape.Remove[S, Idx]]
    protected def run[Idx <: Int](idx: Int) = {
      val res = torch.max(native, idx, false)
      MaxResult(new Tensor(res.get0()), new Tensor(res.get1()))
    }

    /** Variant of this operation that keeps the targeted dimension (reduced to one), rather than removing it. */
    case object keepDim extends DimOperator.Of1[S, T] {
      type Out[Idx <: Int] = MaxResult[Shape.Replace[S, Dim.One, Idx]]
      protected def run[Idx <: Int](idx: Int) = {
        val res = torch.max(native, idx, true)
        MaxResult(new Tensor(res.get0()), new Tensor(res.get1()))
      }
    }
  }

  /** Returns the maximum across a given dimension, along with the indexes where that maximum was found. */
  val maxBy = new MaxReduceOp

  /** Returns a new tensor with the last dimension padded to [dim]. [dim] must be at least as large as the current last dimension. */
  def padTo[D <: Dim, V](dim: D, value: V, mode: PaddingMode = PaddingMode.Append)(using ops: DTypeOps.Scalar[T, V]): Shaped[Shape.Replace[S, D, Shape.LastIdx[S]]] = {
    val n = dim.size - size.last
    if (n == 0) new Tensor(native) else {
      assert(n > 0, s"Can't pad dimension of size ${size.last} to lower size ${dim.size}")

      val padding = mode match {
        case PaddingMode.Append => Array(0L, n)
        case PaddingMode.Prepend => Array(n, 0L)
      }
      new Tensor(torch.pad(native, padding, "constant", new pytorch.DoubleOptional(ops.toDouble(value))))
    }
  }

  /** Returns a new tensor with the last dimension padded to [dim]. If the source is bigger than [dim], returns None. */
  def padToOption[D <: Dim, V](dim: D, value: V, mode: PaddingMode = PaddingMode.Append)(using ops:DTypeOps.Scalar[T, V]): Option[Shaped[Shape.Replace[S, D, Shape.LastIdx[S]]]] = {
    Option.when(dim.size - size.last >= 0)(padTo(dim, value, mode))
  }

  /** Casts the shape of this tensor into compatible shape [O] (which must be a Tuple of Dim's, or a single Dim) */
  def shaped[O](using ev:CanShaped[S, O]): Shaped[ev.Out] = new Tensor(native)

  /** Returns the sizes of all dimensions of the shape of this tensor. */
  def size: Seq[Long] = {
    // Don't use unsafeWrapArray, since the returned array might be freed after returning.
    native.sizes.vec.get.toVector
  }

  /** Returns the size of one dimension selected by D, as a Dim.Ref (since we can't create an actual instance of a Dim). */
  def sizeOf = new DimOperator.Of1[S, T] {
    type Out[Idx <: Int] = Dim.Ref[Elem[S, Idx]]
    def run[Idx <: Int](idx: Int) = Dim.Ref(size(idx))
  }

  /** Returns the size of one dimension typed D, using [dim] to create the instance of D holding the result. */
  def sizeOf[D <: Dim, Idx <: Int](dim: Long => D)(using sel: Shape.SelectIdx[S,D]): D = dim(size(sel.idx))

  val softmax = new DimOperator.Of1Tensor[S, T, D] {
    type Out[Idx <: Int] = S
    def run[Idx <: Int](idx: Int) = new Tensor(native.softmax(idx))
  }

  def sum: Shaped[Scalar] = new Tensor(native.sum())
  val sumBy = new ReduceOp((idx, keep) => native.sum(Array(idx), keep, new ScalarTypeOptional))

  def summary: String = {
    val res = new StringBuilder()
    val isFloat = dtype == DType.bfloat16 || dtype == DType.float16 || dtype == DType.float32 || dtype == DType.float64
    val values = if(isFloat)
      flatten.to(Device.CPU, DType.float32).value
    else
      flatten.to(Device.CPU, DType.int64).value

    val s = size
    if (s.isEmpty) {
      values.head.toString
    } else {
      def indent(i: String, s: String) = s.split("\n").map(l => i + l).mkString("\n")

      def sum[T](sizes: Seq[Long], values: Seq[T]): String = {
        if (sizes.size == 1) {
          if (values.headOption.exists(v => v.isInstanceOf[Float] || v.isInstanceOf[Double])) {
            values.map(o => f"${o.asInstanceOf[Float]}%.4f")mkString("(", ", ", ")")
          } else {
            values.map(o => f"${o.asInstanceOf[Long]}%d")mkString("(", ", ", ")")
          }
        } else {
          values
            .grouped(sizes.drop(1).foldLeft(1L)(_ * _).toInt)
            .map(s => indent("  ", sum(sizes.drop(1), s)))
            .mkString("(\n", ",\n", ")")
        }
      }

      sum(size, values.toSeq)
    }
  }

  /** Converts the tensor to the given device and dtype */
  def to[D1 <: Device, T1 <: DType](device: D1, dtype: T1): Tensor[S, T1, D1] = new Tensor(native.to(device.native, dtype.native))
  /** Converts the tensor to the given device */
  def to[D1 <: Device](device: D1): Tensor[S, T, D1] = new Tensor(native.to(device.native, dtype.native))
  /** Converts the tensor to the given dtype */
  def to[T1 <: DType](dtype: T1): Tensor[S, T1, D] = new Tensor(native.to(dtype.native))

  /** Converts the Tensor to the Default[DType] and Default[Device] in scope. */
  def toDeviceDType[D1 <: Device, T1 <: DType](using t:Default[T1], d:Default[D1]): Tensor[S, T1, D1] = to(d.value, t.value)

  /** Converts the Tensor to the Default[Device] in scope. */
  def toDevice[D1 <: Device](using d:Default[D1]): Tensor[S, T, D1] = to(d.value)

  /** Converts the Tensor to the Default[DType] in scope. */
  def toDType[T1 <: DType](using t:Default[T1]): Tensor[S, T1, D] = to(t.value)

  def tril(diagonal: Long = 0)(using Shape.Size[S] >= 2 =:= true): This = new Tensor(native.tril(diagonal))
  def triu(diagonal: Long = 0)(using Shape.Size[S] >= 2 =:= true): This = new Tensor(native.triu(diagonal))

  /** Swaps the given two dimensions. */
  val transpose = new DimOperator.Of2Tensor[S, T, D] {
    type Out[I1 <: Int, I2 <: Int] = Shape.Swap[S, I1, I2]
    def run[I1 <: Int, I2 <: Int](i1: Int, i2: Int) = new Tensor(native.transpose(i1, i2))
  }

  /** Swaps the last two dimensions. Tensor must have >= 2 dimensions. */
  def t[R <: Tuple](using Transpose[S, R]): Shaped[R] = {
    new Tensor(native.transpose(-2L, -1L))
  }

  /** Returns a view of this Tensor with just "Dim" as type for each
    * dimension. This makes it easier to create collections of
    * same-dimension but different length tensors. */
  def untyped(using ut: Untyped[S]): Shaped[ut.Out] = new Tensor(native)

  /** Returns a view of this Tensor with 0 dimensions, or None if the tensor has a different number of dimensions. */
  def untyped0D: Option[Shaped[EmptyTuple]] = Option.when(size.length == 0)(new Tensor(native))

  /** Returns a view of this Tensor with 1 dimension, or None if the tensor has a different number of dimensions. */
  def untyped1D: Option[Shaped[Dim *: EmptyTuple]] = Option.when(size.length == 1)(new Tensor(native))

  /** Returns a view of this Tensor with 2 dimensions, or None if the tensor has a different number of dimensions. */
  def untyped2D: Option[Shaped[(Dim, Dim)]] = Option.when(size.length == 2)(new Tensor(native))

  /** Returns a view of this Tensor with 3 dimensions, or None if the tensor has a different number of dimensions. */
  def untyped3D: Option[Shaped[(Dim, Dim, Dim)]] = Option.when(size.length == 3)(new Tensor(native))

  /** Returns a view of this Tensor with 4 dimensions, or None if the tensor has a different number of dimensions. */
  def untyped4D: Option[Shaped[(Dim, Dim, Dim, Dim)]] = Option.when(size.length == 4)(new Tensor(native))

  /** Returns a view of this Tensor with 5 dimensions, or None if the tensor has a different number of dimensions. */
  def untyped5D: Option[Shaped[(Dim, Dim, Dim, Dim, Dim)]] = Option.when(size.length == 5)(new Tensor(native))

  /** Inserts a dimension of One after D */
  val unsqueezeAfter = new DimOperator.Of1Tensor[S, T, D] {
    type Out[Idx <: Int] = Shape.InsertAfter[S, Dim.One, Idx]
    def run[Idx <: Int](idx: Int) = new Tensor(native.unsqueeze(idx + 1))
  }

  /** Inserts a dimension of One before D */
  val unsqueezeBefore = new DimOperator.Of1Tensor[S, T, D] {
    type Out[Idx <: Int] = Shape.InsertBefore[S, Dim.One, Idx]
    def run[Idx <: Int](idx: Int) = new Tensor(native.unsqueeze(idx))
  }

  /** Provides alternative views to this Tensor, without changing the underlying storage. */
  case object view {
    /** Transforms a split version of this tensor, split across dimension D in N parts. */
    val split = new DimOperator.Of1[S, T] {
      type Out[Idx <: Int] = SplitApply[Idx, Elem[S, Idx]]
      def run[Idx <: Int](idx: Int) = new SplitApply(idx)
    }
    class SplitApply[Idx <: Int, D](private[s3torch] val idx: Int) {
      /** Splits the selected dimension into N parts, i.e. the dimension D gets split into two dimensions (N, D / N) */
      def into[N <: Dim](n: N)(using ev:Split[D, N]): Shaped[Shape.ReplaceWithTuple[S, ev.Out, Idx]] = {
        val (before, after) = size.splitAt(idx)
        val dimsize = after.head
        val sizes = before :+ n.size :+ (dimsize / n.size) :++ after.tail
        new Tensor(native.view(sizes.toArray*))
      }
    }

    // "merge" can't be a DimOperator, since we simultaneously need to find the index, and verify which Unsplit type to apply.
    private type Merge[I <: Int] = Unsplit[Elem[S, I - 1], Elem[S, I]]
    private type Merged[I <: Int, D <: Dim] = Shape.Replace2WithTuple[S, Tuple1[D], I]
    /** Merges the selected dimension with the one before it, by
      * multiplying the dimensions. If these have been previously split using
      * split(), this operation performs the reverse. */
    def merge[D](using sel: Shape.SelectIdx[S, D], ev: Merge[sel.Idx]): Shaped[Merged[sel.Idx, ev.Out]] = {
      val (before, after) = size.splitAt(sel.idx - 1)
      val sizes = before :+ (after(0) * after(1)) :++ after.drop(2)
      new Tensor(native.view(sizes.toArray*))
    }
    /** Merges the selected dimension with the one before it, by
      * multiplying the dimensions. If these have been previously split using
      * split(), this operation performs the reverse. */
    def merge[D](d: D)(using sel: Shape.SelectIdx[S, D], ev: Merge[sel.Idx]): Shaped[Merged[sel.Idx, ev.Out]] = merge[D]
  }

  def value[O](using ev: TensorValue[S, T, O])(using D =:= CPU.type): O = ev(native)
  def value_=[V](v: V)(using updateSource: UpdateSource[V, D]): Unit = {
    updateSource(native, Tensor.tensorIndexArray(), v)
  }

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

  def isNan: Tensor[S, DType.Bool, D] = new Tensor(native.isnan())

  private[Tensor] def unsafeWithShape[S1 <: Tuple]: Shaped[S1] = this.asInstanceOf
}

/** Math functions like sin, exp, are definied here, since "sin(x)"
  * approximated mathemetical notation better than "x.sin", even
  * though the latter would be more idiomatic Scala. */
object Tensor {
  // TODO nicer .apply:
  // - Reshaping or creating with known dimension:
  //   Tensor.shaped(Tuple(AnyDim))((Values))     // return option if too big?
  //   Tensor.shaped[Tuple(StaticDim)]((Values))  // return option if too big?
  //   Tensor.shaped[Tuple(StaticDim)]((tuple))   // staticaly checked
  //   OR just do this in .shaped instance method.

  /** Creates a Tensor from the given value, which can be a scalar, a
    * (potentially) nested Seq, or a (potentially nested) tuple,
    * picking an appropriate shape and DType. */
  def apply[V, T <: DType, D <: Device](value: V)(using t: DTypeFor[TensorApply.BaseType[V]], device: Default[D], ev:TensorApply[V]): Tensor[ev.OutShape, t.Out, D] = {
    new Tensor(ev(value, device.value))
  }

  /** Creates a range, returning a DType that follows the Default[DType] */
  def arangeOfD[D <: Dim, T <: DType, Dv <: Device](dim: D)(using device: Default[Dv], dType: Default[T], ops: DTypeOps.Scalar[T, ?]): Tensor[Tuple1[D], T, Dv] = arangeD(0L, dim.size, 1L)(using ops = DTypeOps.ScalarFromLong(ops)).unsafeWithShape

  /** Creates a range of Int64 DType (since dimensions are Long) */
  // TODO: For static din, auto-pick Int32, Int16 or Int8 for lower values.
  def arangeOf[D <: Dim, Dv <: Device](dim: D)(using Default[Dv]): Tensor[Tuple1[D], Int64, Dv] = arange(0L, dim.size, 1L).unsafeWithShape

  /** Creates a range, returning a DType that follows the Default[DType] */
  def arangeD[V, T <: DType, Dv <: Device](start: V, end: V, step: V)(using dType: Default[T], ops: DTypeOps.Scalar[T, V], dv: Default[Dv]): Tensor[Tuple1[Dim.Dynamic], T, Dv] = {
    new Tensor(torch.torch_arange(ops.fromScalar(start), ops.fromScalar(end), ops.fromScalar(step), Torch.tensorOptions(dType.value, dv.value)))
  }
  /** Creates a range, returning a DType that that is a good match for V */
  def arange[V, Dv <: Device](start: V, end: V, step: V)(using t: DTypeFor[V], ops: DTypeOps.Scalar[t.Out, V], dv: Default[Dv]): Tensor[Tuple1[Dim.Dynamic], t.Out, Dv] = {
    new Tensor(torch.torch_arange(ops.fromScalar(start), ops.fromScalar(end), ops.fromScalar(step), Torch.tensorOptions(t.dType, dv.value)))
  }

  // TODO consider a FunctionApply abstraction, to clean up duplication here
  def cos[S <: Tuple, T <: DType, D <: Device](t: Tensor[S, T, D]): Tensor[S, T, D] = new Tensor(t.native.cos)
  def cosh[S <: Tuple, T <: DType, D <: Device](t: Tensor[S, T, D]): Tensor[S, T, D] = new Tensor(t.native.cosh)
  def exp[S <: Tuple, T <: DType, D <: Device](t: Tensor[S, T, D]): Tensor[S, T, D] = new Tensor(t.native.exp)
  def relu[S <: Tuple, T <: DType, D <: Device](t: Tensor[S, T, D]): Tensor[S, T, D] = new Tensor(t.native.relu)
  def sin[S <: Tuple, T <: DType, D <: Device](t: Tensor[S, T, D]): Tensor[S, T, D] = new Tensor(t.native.sin)
  def sinh[S <: Tuple, T <: DType, D <: Device](t: Tensor[S, T, D]): Tensor[S, T, D] = new Tensor(t.native.sinh)
  def tan[S <: Tuple, T <: DType, D <: Device](t: Tensor[S, T, D]): Tensor[S, T, D] = new Tensor(t.native.tan)
  def tanh[S <: Tuple, T <: DType, D <: Device](t: Tensor[S, T, D]): Tensor[S, T, D] = new Tensor(t.native.tanh)

  /** Returns a tensor filled with uninitialized data. */
  def empty[T <: DType, D <: Device](using dtype: Default[T], device: Default[D]) =
    new ZerosApply(dtype.value, device.value, torch.torch_empty(_, _, new pytorch.MemoryFormatOptional))
  def full[D <: Device, V](value: V)(using t: DTypeFor[V], device: Default[D], ops: DTypeOps.Scalar[t.Out, V]) =
    new ZerosApply(t.dType, device.value, torch.full(_, ops.fromScalar(value), _))
  def fullD[T <: DType, D <: Device, V](value: V)(using dtype: Default[T], device: Default[D], ops: DTypeOps.Scalar[T, V]) =
    new ZerosApply(dtype.value, device.value, torch.full(_, ops.fromScalar(value), _))
  def ones[T <: DType, D <: Device](using dtype: Default[T], device: Default[D]) =
    new ZerosApply(dtype.value, device.value, torch.torch_ones(_, _))
  def rand[T <: DType, D <: Device](using dtype: Default[T], device: Default[D], rnd:RandomSource) =
    rnd(new ZerosApply(dtype.value, device.value, torch.torch_rand(_, _)))
  /** Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive), as Int64 */
  def randint[D <: Device](low: Long, high: Long)(using device: Default[D], rnd:RandomSource): ZerosApply[Int64, D] =
    rnd(new ZerosApply(int64, device.value, torch.torch_randint(low, high, _, _)))
  /** Returns a tensor filled with random integers generated uniformly between 0 (inclusive) and high (exclusive), as Int64. */
  def randint[D <: Device](high: Long)(using device: Default[D], rnd:RandomSource): ZerosApply[Int64, D] = randint(0, high)
  /** Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive), which follows the Default[DType]. */
  def randintD[T <: DType, D <: Device](low: Long, high: Long)(using dtype: Default[T], device: Default[D], rnd:RandomSource): ZerosApply[T, D] =
    rnd(new ZerosApply(dtype.value, device.value, torch.torch_randint(low, high, _, _)))
  /** Returns a tensor filled with random integers generated uniformly between 0 (inclusive) and high (exclusive), which follows the Default[DType]. */
  def randintD[T <: DType, D <: Device](high: Long)(using dtype: Default[T], device: Default[D], rnd:RandomSource): ZerosApply[T, D] = randintD(0, high)
  def randperm[T <: DType, D <: Device, N <: Dim](dim: N)(using dtype: Default[T], device: Default[D]): Tensor[Tuple1[N], T, D] =
    new Tensor(torch.torch_randperm(dim.size, Torch.tensorOptions(dtype.value, device.value)))
  def zeros[T <: DType, D <: Device](using dtype: Default[T], device: Default[D]) =
    new ZerosApply(dtype.value, device.value, torch.torch_zeros(_, _))

  /** Concatenates a sequence of tensors along a new dimension. */
  def stack[B <: Dim] = new StackApply[B](None)
  def stack[B <: Dim](batchDim: B) = new StackApply[B](Some(batchDim.size.toInt))
  class StackApply[B <: Dim](expected: Option[Int]) {
    def apply[S <: Tuple, T <: DType, D <: Device](tensors: Iterable[Tensor[S, T, D]]): Tensor[B *: S, T, D] = {
      if (expected.exists(c => tensors.size != c)) {
        throw new IllegalArgumentException(s"Expected ${expected.get} tensors, but got ${tensors.size}")
      }
      new Tensor(torch.stack(new pytorch.TensorVector(tensors.map(_.native).toArray*)))
    }
  }

  /** Runs the given block with gradients disabled. All computations will be performed as if having set requiresGrad == false, even if true was passed. */
  def noGrad[T](block: =>T): T = {
    Using.resource(new pytorch.NoGradGuard)(_ => block)
  }

  // ---- Methods on Tensor that require floats
  extension[S <: Shape, T <: DType.Floaty, Dv <: Device](t: Tensor[S, T, Dv]) {
    def stdBy = new t.ReduceOp((idx, keep) => t.native.std(Array(idx), new pytorch.ScalarOptional, keep))
    def stdBy(correction: Double) = new t.ReduceOp((idx, keep) => t.native.std(Array(idx), new pytorch.ScalarOptional(new pytorch.Scalar(correction)), keep))
    def meanBy = new t.ReduceOp((idx, keep) => t.native.mean(Array(idx), keep, new ScalarTypeOptional))

  }

  // ---- Methods on Tensor that only exist on scalars
  extension[T <: DType, D <: Device](t: Tensor[Scalar, T, D]) {
    def backward(): Unit = t.native.backward()

    /** Adds a single dimension to this scalar, turning it into a vector of size one. */
    def unsqueeze: t.Shaped[Tuple1[Dim.One]] = new Tensor(t.native.unsqueeze(0))
  }

  // ---- Methods on Tensor with at least 1 dimension
  extension[T <: DType, D <: Device, D1 <: Dim, DT <: Tuple](t: Tensor[D1 *: DT, T, D]) {
    def ++[U <: Tuple](that: t.Shaped[U])(using Cat[D1 *: DT, U, 0])(using pick:Cat.PickDynamic[D1, Tuple.Head[U]]): t.Shaped[pick.Out *: DT] =
      new Tensor(torch.cat(new pytorch.TensorVector(t.native, that.native), 0))
  }

  // ---- Methods on Tensor with 1 dimension ---
  extension[T <: DType, D <: Device, D1 <: Dim](t: Tensor[Tuple1[D1], T, D]) {
    def apply[I1 <: Index](v1: I1)(using i1: Index.Valid[D1, I1]): t.Shaped[i1.Apply] = {
      new Tensor(t.native.index(new pytorch.TensorIndexVector(v1.toNative)))
    }

    def update[I1 <: Index, V](i: I1, value: V)(using Index.Valid[D1, I1])(using updateSource: UpdateSource[V, D]): Unit = {
      updateSource(t.native, tensorIndexArray(i), value)
    }
  }

  // ---- Methods on Tensor with 2 dimensions ---
  extension[T <: DType, D <: Device, D1 <: Dim, D2 <: Dim](t: Tensor[(D1, D2), T, D]) {
    def apply[I1 <: Index, I2 <: Index](v1: I1, v2: I2)(
      using i1: Index.Valid[D1, I1], i2: Index.Valid[D2, I2]
    ): t.Shaped[i1.Apply ++ i2.Apply] = {
      new Tensor(t.native.index(new pytorch.TensorIndexVector(v1.toNative, v2.toNative)))
    }

    def update[I1 <: Index, I2 <: Index, V](i: (I1, I2), value: V)(
      using Index.Valid[D1, I1], Index.Valid[D2, I2]
    )(
      using updateSource: UpdateSource[V, D]
    ): Unit = {
      updateSource(t.native, tensorIndexArray(i._1, i._2), value)
    }
  }

  // ---- Methods on Tensor with 3 dimensions ---
  extension[T <: DType, D <: Device, D1 <: Dim, D2 <: Dim, D3 <: Dim](t: Tensor[(D1, D2, D3), T, D]) {
    def apply[I1 <: Index, I2 <: Index, I3 <: Index](v1: I1, v2: I2, v3: I3)(
      using i1: Index.Valid[D1, I1], i2: Index.Valid[D2, I2], i3: Index.Valid[D3, I3]
    ): t.Shaped[i1.Apply ++ i2.Apply ++ i3.Apply] = {
      new Tensor(t.native.index(new pytorch.TensorIndexVector(v1.toNative, v2.toNative, v3.toNative)))
    }

    def update[I1 <: Index, I2 <: Index, I3 <: Index, V](i: (I1, I2, I3), value: V)(
      using Index.Valid[D1, I1], Index.Valid[D2, I2], Index.Valid[D3, I3]
    )(
      using updateSource: UpdateSource[V, D]
    ): Unit = {
      updateSource(t.native, tensorIndexArray(i._1, i._2, i._3), value)
    }
  }

  private[Tensor] def tensorIndexArray(i: Index*) = pytorch.TensorIndexArrayRef (new pytorch.TensorIndexVector(i.map(_.toNative)*))
}
