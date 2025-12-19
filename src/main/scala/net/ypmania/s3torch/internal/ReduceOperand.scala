package net.ypmania.s3torch.internal

import org.bytedeco.pytorch

import net.ypmania.s3torch.Shape
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Dim

trait ReduceOperand[S <: Shape, D, Idx <: Int, K <: ReduceOperand.Variant] {
  type Out <: Shape
  def index: Long
  def keep: Boolean
}

object ReduceOperand {
  sealed trait Variant
  case object Reduce extends Variant
  case object KeepDim extends Variant

  object Variant {
    given defaultReduce: Reduce.type = Reduce
  }

  given [S <: Shape, D, Idx <: Int](using sel: Shape.Select[S,D,Idx], idx: ValueOf[Idx]): ReduceOperand[S, D, Idx, Reduce.type] with {
    type Out = Shape.Remove[S, Idx]
    def index = idx.value
    def keep = false
  }

  given [S <: Shape, D, Idx <: Int](using sel: Shape.Select[S,D,Idx], idx: ValueOf[Idx]): ReduceOperand[S, D, Idx, KeepDim.type] with {
    type Out = Shape.Replace[S, Dim.One, Idx]
    def index = idx.value
    def keep = true
  }
}
