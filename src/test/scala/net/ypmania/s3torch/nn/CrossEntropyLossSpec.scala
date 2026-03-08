package net.ypmania.s3torch.nn

import net.ypmania.s3torch.UnitSpec
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Device.CPU
import net.ypmania.s3torch.DType.Float64

class CrossEntropyLossSpec extends UnitSpec {
  describe("CrossEntropyLoss") {
    val crossEntropy = CrossEntropyLoss.indexesReduce()

    it("should apply indexes to a target") {
      val inputs = Tensor((
        ((0.1, 0.7, 0.2)),
        ((0.9, 0.1, 0.0))
      ))
      val targets = Tensor((1L, 0L))
      val res = crossEntropy(inputs, targets)
      val resType: Tensor[EmptyTuple.type, Float64, CPU.type] = res
      assert(res.value === 0.6931)
    }
  }
}
