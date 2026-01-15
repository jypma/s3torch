package net.ypmania.s3torch.nn

import net.ypmania.s3torch.UnitSpec
import net.ypmania.s3torch.Dim.Static
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.DType.Float32

class LinearSpec extends UnitSpec {
  case object In extends Static[2L]
  case object Out extends Static[3L]
  case object BatchSize extends Static[1L]

  describe("Linear") {
    val lin = withSeed(0) { Linear(In, Out) }

    it("Turns the last dimension from In to Out") {
      val in = Tensor.zeros(BatchSize, In)
      val out = lin(in)
      val outType: Tensor[(BatchSize.type, Out.type), Float32.type] = out
      assert(out.value === Seq(Seq(-0.0140, 0.5606, -0.0627)))
    }
  }
}
