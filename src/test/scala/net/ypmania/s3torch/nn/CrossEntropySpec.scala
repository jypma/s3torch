package net.ypmania.s3torch.nn

import net.ypmania.s3torch.UnitSpec
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Device.CPU
import net.ypmania.s3torch.DType.Float64
import net.ypmania.s3torch.DType.Float16
import net.ypmania.s3torch.DType

class CrossEntropySpec extends UnitSpec {
  describe("CrossEntropy") {
    it("should apply indexes to a target") {
      val inputs = Tensor((
        ((0.1, 0.7, 0.2)),
        ((0.9, 0.1, 0.0))
      ))
      val targets = Tensor((1L, 0L))
      val res = CrossEntropy(inputs, targets)
      val resType: Tensor[EmptyTuple.type, Float64, CPU.type] = res
      assert(res.value === 0.6931)
    }

    it("should preserve its input DType as output") {
      val inputs = Tensor((
        ((0.1, 0.7, 0.2)),
        ((0.9, 0.1, 0.0))
      )).to(DType.float16)
      val targets = Tensor((1L, 0L))
      val res = CrossEntropy(inputs, targets)
      assert(res.dtype == DType.float16)
      val resType: Tensor[EmptyTuple.type, Float16, CPU.type] = res
      assert(res.value === 0.6933)
    }
  }
}
