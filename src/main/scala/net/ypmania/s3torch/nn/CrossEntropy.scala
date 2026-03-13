package net.ypmania.s3torch.nn

import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Shape.Scalar
import net.ypmania.s3torch.Tensor
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch

object CrossEntropy {
  /** Cross entropy where the targets are indexes, and which reduces the loss to a single scalar value.
    * Pytorch only supports Int64 as the target tensor. */
  def apply[
    SI <: Tuple, ST <: Tuple, D <: Device, TI <: DType, TT <: DType.Int64
  ] (
    input: Tensor[SI, TI, D], target: Tensor[ST, TT, D],
    ignoreIndex: Option[Long] = None, reduction: Reduction = Reduction.Mean, labelSmoothing: Double = 0
  ) (using
    ValidShape[SI, ST]
  ): input.Shaped[Scalar] = {

    val opts = new pytorch.CrossEntropyLossOptions
    opts.label_smoothing().put(labelSmoothing)
    ignoreIndex.foreach(opts.ignore_index().put)
    reduction match {
      case Reduction.Mean => opts.reduction().put(new pytorch.kMean())
      case Reduction.Sum => opts.reduction().put(new pytorch.kSum())
    }
    new Tensor(torch.cross_entropy(input.native, target.native, opts))
  }

  enum Reduction {
    case Mean extends Reduction
    case Sum extends Reduction
  }

  trait ValidShape[I <: Tuple, T <: Tuple]
  // Rules from https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#crossentropyloss
  object ValidShape {
    given s1[C <: Dim]: ValidShape[C *: EmptyTuple, EmptyTuple] with {}
    given s2[N <: Dim, C <: Dim, Tail <: Tuple]: ValidShape[(N *: C *: Tail), N *: Tail] with {}
  }
}
