package net.ypmania.s3torch.internal

import org.bytedeco.pytorch

import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Shape
import net.ypmania.s3torch.Shape.Concat
import net.ypmania.s3torch.SelectIdx
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Batched

/** A collection of operators that allow (a) dimension(s) to be selected using the Select.* traits, and then perform further
  * compile-time transformations on a Tensor's shape, depending on which dimensions were selected. */
class DimOperator[S <: Shape, T <: DType, Dv <: Device] {
  /** A method that operates on a single dimension, which can be selected using one of the Select traits, either as a type or as a
    * value. It returns a non-Tensor type. */
  abstract class Of1Ev[E[_ <: Shape, _ <: Int]] {
    type Out[B <: Shape, L <: Shape, Idx <: Int]
    protected def runE[B <: Shape, L <: Shape, Idx <: Int](idx: Int)(using ev: E[L, Idx]): Out[B, L, Idx]

    def apply[B <: Shape, L <: Shape, D](d: D)(using
      b: Batched[B, L, S],
      sel: SelectIdx[L, D],
      ev: E[L, sel.Idx]
    ): Out[B, L, sel.Idx] = runE(sel.idx + b.batchSize.value)
  }

  /** A method that operates on a single dimension, which can be selected using one of the Select traits, either as a type or as a
    * value. It returns a non-Tensor type. */
  abstract class Of1 extends Of1Ev[[L <: Tuple, Idx <: Int] =>> NotNeeded]{
    protected def run[B <: Shape, L <: Shape, Idx <: Int](idx: Int): Out[B, L, Idx]
    override final protected def runE[B <: Shape, L <: Shape, Idx <: Int](idx: Int)(using ev: NotNeeded) = run(idx)
  }

  /** A method that operates on a single dimension, which can be selected using one of the Select traits, either as a type or as a
    * value. It returns a Tensor. */
  abstract class Of1Tensor extends Of1 {
    type OutT[L <: Shape, Idx <: Int] <: Shape
    type Out[B <: Shape, L <: Shape, Idx <: Int] = Tensor[Concat[B, OutT[L, Idx]], T, Dv]

    override final protected def run[B <: Shape, L <: Shape, Idx <: Int](idx: Int) = new Tensor(runT(idx))

    protected def runT(idx: Int): pytorch.Tensor
  }

  /** A method that operates on a single dimension, which can be
    * selected using one of the Select traits, either as a type
    * or as a value. It returns a Tensor. */
  abstract class Of1TensorOld[S <: Shape, T <: DType, Dv <: Device] {
    type Out[L <: Shape, Idx <: Int] <: Shape
    protected def run(idx: Int): pytorch.Tensor

    def apply[B <: Shape, L <: Shape, D](d: D)(using
      b: Batched[B, L, S],
      sel: SelectIdx[L, D],
      v: VerifyShape[Out[L, sel.Idx]]
    ): Tensor[Concat[B, Out[L, sel.Idx]], T, Dv] = new Tensor(run(sel.idx + b.batchSize.value))
  }

  /** Variant of OfTensor that additionally requires a given of type E, depending on the selected dimension. */
  abstract class Of1TensorEv[E[_ <: Shape, _ <: Int]] extends Of1Ev[E] {
    type OutT[L <: Shape, Idx <: Int] <: Shape
    type Out[B <: Shape, L <: Shape, Idx <: Int] = Tensor[Concat[B, OutT[L, Idx]], T, Dv]

    override final protected def runE[B <: Shape, L <: Shape, Idx <: Int](idx: Int)(using ev: E[L, Idx]) = new Tensor(runT(idx))

    protected def runT[L <: Shape, Idx <: Int](idx: Int)(using ev: E[L, Idx]): pytorch.Tensor
  }

  /** A method that operates on two dimensions, which can be selected using one of the Select traits, either as a type or as a
    * value. It returns a Tensor. */
  abstract class Of2Tensor {
    type Out[L <: Shape, I1 <: Int, I2 <: Int] <: Shape
    protected def run[L <: Shape, I1 <: Int, I2 <: Int](i1: Int, i2: Int): pytorch.Tensor

    def apply[B <: Shape, L <: Shape, D1, D2](d1: D1, d2: D2)(using
      b: Batched[B, L, S],
      s1: SelectIdx[L, D1],
      s2: SelectIdx[L, D2],
      v: VerifyShape[Out[L, s1.Idx, s2.Idx]]
    ): Tensor[Concat[B, Out[L, s1.Idx, s2.Idx]], T, Dv] = new Tensor(run(s1.idx + b.batchSize.value, s2.idx + b.batchSize.value))
  }
}

/** A given that indicates nothing is needed here, which always has an instance available. */
trait NotNeeded
object NotNeeded {
  given notNeeded: NotNeeded = new NotNeeded {}
}
