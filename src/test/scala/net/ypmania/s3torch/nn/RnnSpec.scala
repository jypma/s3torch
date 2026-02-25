package net.ypmania.s3torch.nn

import net.ypmania.s3torch.UnitSpec
import net.ypmania.s3torch.Dim.Static
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Device.CPU
import net.ypmania.s3torch.DType.Int32
import net.ypmania.s3torch.nn.rnn.PaddingSide

class RnnSpec extends UnitSpec {
  describe("batchPadSequences") {
    it("pads different-length sequences to match given sequence length") {
      case object BatchSize extends Static[3L]
      case object SequenceLength extends Static[5L]

      val r = rnn.batchPadSequences(BatchSize, SequenceLength, Seq(
        Tensor((1, 2)).untyped,
        Tensor((1, 2, 3)).untyped,
        Tensor((1, 2, 3, 4)).untyped
       ), 0, PaddingSide.Right)
      val rType: Tensor[(BatchSize.type, SequenceLength.type), Int32.type, CPU.type] = r

      assert(r.value.toSeq === Seq(
        Seq(1, 2, 0, 0, 0),
        Seq(1, 2, 3, 0, 0),
        Seq(1, 2, 3, 4, 0))
      )
    }
  }
}
