package net.ypmania.s3torch.nn

import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.PaddingMode
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.internal.Torch
import net.ypmania.s3torch.internal.FromScala.ToScalar

object rnn {
  /** Pads a batch of given sequences to the given length. The following must be true at runtime:
    * - [sequences] must have exactly [BatchSize] elements
    * - Each sequence must have [SequenceLength] or fewer elements
    */
  def batchPadSequences[T <: DType, Dv <: Device, BatchSize <: Dim, SequenceLength <: Dim]
    (batchSize: BatchSize, sequenceLength: SequenceLength, sequences: Seq[Tensor[Tuple1[? <: Dim], T, Dv]], paddingValue: Double, mode: PaddingMode)
      : Tensor[(BatchSize, SequenceLength), T, Dv] = {
    require(sequences.size == batchSize.size.toInt) // Also verifies that size > 0

    val toScalar = summon[ToScalar[Double]]
    val firstPadding = torch.full(Array(sequenceLength.size - sequences.head.size(0)), toScalar(paddingValue), Torch.tensorOptions(sequences.head.dtype, sequences.head.device))
    val first = mode match {
      case PaddingMode.Prepend => torch.concat(new pytorch.TensorVector(firstPadding, sequences.head.native))
      case PaddingMode.Append => torch.concat(new pytorch.TensorVector(sequences.head.native, firstPadding))
    }
    val tensorsIn = first +: sequences.tail.map(_.native)
    new Tensor(torch.pad_sequence(new pytorch.TensorVector(tensorsIn*), true, paddingValue, paddingSide(mode)))
  }

  /** Pads a batch of given sequences to the given length. The following must be true at runtime:
    * - Each sequence must have [SequenceLength] or fewer elements
    */
  def batchPadSequences[T <: DType, Dv <: Device, SequenceLength <: Dim]
    (sequenceLength: SequenceLength, sequences: Seq[Tensor[Tuple1[? <: Dim], T, Dv]], paddingValue: Double, mode: PaddingMode)
      : Tensor[(Dynamic, SequenceLength), T, Dv] = {
    new Tensor(torch.pad_sequence(new pytorch.TensorVector(sequences.map(_.native)*), true, paddingValue, paddingSide(mode)))
  }

  private def paddingSide(mode: PaddingMode) = mode match {
    case PaddingMode.Prepend => "left"
    case PaddingMode.Append => "right"
  }
}
