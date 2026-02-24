package net.ypmania.s3torch.nn

import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch
import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Dim

object rnn {
  enum PaddingSide(private[rnn] val native: String) {
    case Left extends PaddingSide("left")
    case Right extends PaddingSide("right")
  }

  /** Pads a batch of given sequences to the given length. The following must be true at runtime:
    * - [sequences] must have exactly [BatchSize] elements
    * - Each sequence must have [SequenceLength] or fewer elements
    */
  def batchPadSequences[T <: DType, Dv <: Device, BatchSize <: Dim, SequenceLength <: Dim]
    (batchSize: BatchSize, sequenceLength: SequenceLength, sequences: Seq[Tensor[Tuple1[? <: Dim], T, Dv]], paddingValue: Double, paddingSide: PaddingSide)
      : Tensor[(BatchSize, SequenceLength), T, Dv] = {
    require(sequences.size == batchSize.size.toInt)

    new Tensor(torch.pad_sequence(new pytorch.TensorVector(sequences.map(_.native)*), true, paddingValue, paddingSide.native))
  }

  /** Pads a batch of given sequences to the given length. The following must be true at runtime:
    * - Each sequence must have [SequenceLength] or fewer elements
    */
  def batchPadSequences[T <: DType, Dv <: Device, SequenceLength <: Dim]
    (sequenceLength: SequenceLength, sequences: Seq[Tensor[Tuple1[? <: Dim], T, Dv]], paddingValue: Double, paddingSide: PaddingSide)
      : Tensor[(Dynamic, SequenceLength), T, Dv] = {
    new Tensor(torch.pad_sequence(new pytorch.TensorVector(sequences.map(_.native)*), true, paddingValue, paddingSide.native))
  }
}
