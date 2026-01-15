package net.ypmania.s3torch.nn

import net.ypmania.s3torch.UnitSpec
import net.ypmania.s3torch.Tensor

class DropoutSpec extends UnitSpec{
  describe("Dropout") {
    val dropout = Dropout(0.5)

    it("should drop values about 50%") {
      val in = Tensor.ones(10L)
      val out = dropout(in)
      assert(out.value === Seq(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0))
    }
  }
}
