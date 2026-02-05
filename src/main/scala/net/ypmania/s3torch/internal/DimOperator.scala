package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Shape
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Device

object DimOperator {
  /** A method that operates on a single dimension, which can be
    * selected using one of the Shape.Select traits, either as a type
    * or as a value. It returns a non-Tensor type. */
  abstract class Of1[S <: Shape, T <: DType] {
    type Out[Idx <: Int]
    protected def run[Idx <: Int](idx: Idx): Out[Idx]

    def apply[D](d: D)(using sel: Shape.SelectIdx[S,D]): Out[sel.Idx] = run(sel.idx)
    def apply[D](using sel: Shape.SelectIdx[S,D]): Out[sel.Idx] = run(sel.idx)
  }

  /** A method that operates on a single dimension, which can be
    * selected using one of the Shape.Select traits, either as a type
    * or as a value. It returns a Tensor. */
  abstract class Of1Tensor[S <: Shape, T <: DType, Dv <: Device] {
    type Out[Idx <: Int] <: Shape
    protected def run[Idx <: Int](idx: Idx): Tensor[Out[Idx], T, Dv]

    def apply[D](d: D)(using sel: Shape.SelectIdx[S,D], v: VerifyShape[Out[sel.Idx]]): Tensor[Out[sel.Idx], T, Dv] = run(sel.idx)
    def apply[D](using sel: Shape.SelectIdx[S,D], v: VerifyShape[Out[sel.Idx]]): Tensor[Out[sel.Idx], T, Dv] = run(sel.idx)
  }

  /** A method that operates on two dimensions, which can be
    * selected using one of the Shape.Select traits, either as a type
    * or as a value. It returns a Tensor. */
  abstract class Of2Tensor[S <: Shape, T <: DType, Dv <: Device] {
    type Out[I1 <: Int, I2 <: Int] <: Shape
    protected def run[I1 <: Int, I2 <: Int](i1: I1, i2: I2): Tensor[Out[I1, I2], T, Dv]

    def apply[D1, D2](d1: D1, d2: D2)(using s1: Shape.SelectIdx[S, D1], s2: Shape.SelectIdx[S, D2], v: VerifyShape[Out[s1.Idx, s2.Idx]]): Tensor[Out[s1.Idx, s2.Idx], T, Dv] = run(s1.idx, s2.idx)
    def apply[D1, D2](using s1: Shape.SelectIdx[S, D1], s2: Shape.SelectIdx[S, D2], v: VerifyShape[Out[s1.Idx, s2.Idx]]): Tensor[Out[s1.Idx, s2.Idx], T, Dv] = run(s1.idx, s2.idx)
  }
}
