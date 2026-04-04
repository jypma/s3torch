package net.ypmania.s3torch.nn

import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.PaddingMode
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.internal.Torch
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch
import net.ypmania.s3torch.DTypeOps

object rnn {
  /** Pads a batch of given sequences to the given length. The following must be true at runtime:
    * - [sequences] must have exactly [BatchSize] elements
    * - Each sequence must have [SequenceLength] or fewer elements
    */
  def batchPadSequences[T <: DType, Dv <: Device, BatchSize <: Dim, SequenceLength <: Dim, V]
  (batchSize: BatchSize, sequenceLength: SequenceLength, sequences: Seq[Tensor[Tuple1[? <: Dim], T, Dv]], paddingValue: V, mode: PaddingMode)
  (using ops: DTypeOps.Scalar[T, V])
      : Tensor[(BatchSize, SequenceLength), T, Dv] = {
    require(sequences.size == batchSize.size.toInt) // Also verifies that size > 0

    val firstPadding = torch.full(Array(sequenceLength.size - sequences.head.size(0)), ops.fromScalar(paddingValue), Torch.tensorOptions(sequences.head.dtype, sequences.head.device))
    val first = mode match {
      case PaddingMode.Prepend => torch.concat(new pytorch.TensorVector(firstPadding, sequences.head.native))
      case PaddingMode.Append => torch.concat(new pytorch.TensorVector(sequences.head.native, firstPadding))
    }
    val tensorsIn = first +: sequences.tail.map(_.native)
    new Tensor(torch.pad_sequence(new pytorch.TensorVector(tensorsIn*), true, ops.toDouble(paddingValue), paddingSide(mode)))
  }

  /** Pads a batch of given sequences to the given length. The following must be true at runtime:
    * - Each sequence must have [SequenceLength] or fewer elements
    */
  def batchPadSequences[T <: DType, Dv <: Device, SequenceLength <: Dim, V]
    (sequenceLength: SequenceLength, sequences: Seq[Tensor[Tuple1[? <: Dim], T, Dv]], paddingValue: V, mode: PaddingMode)
    (using ops: DTypeOps.Scalar[T, V])
      : Tensor[(Dynamic, SequenceLength), T, Dv] = {
    new Tensor(torch.pad_sequence(new pytorch.TensorVector(sequences.map(_.native)*), true, ops.toDouble(paddingValue), paddingSide(mode)))
  }

  private def paddingSide(mode: PaddingMode) = mode match {
    case PaddingMode.Prepend => "left"
    case PaddingMode.Append => "right"
  }
}
