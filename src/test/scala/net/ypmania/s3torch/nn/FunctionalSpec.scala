package net.ypmania.s3torch.nn

import net.ypmania.s3torch.UnitSpec
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.DType.Float32
import net.ypmania.s3torch.Device.CPU

class FunctionalSpec extends UnitSpec {
  describe("scaledDotProductAttention") {
    it("should calculate for minimal tensors") {
      case object L extends Dim.Static[1L]
      case object S extends Dim.Static[2L]
      case object E extends Dim.Static[3L]

      val q = Tensor.rand(L, E)
      val k = Tensor.rand(S, E)
      val v = Tensor.rand(S, E)

      val res = functional.scaledDotProductAttention(q, k, v)
      val resType: Tensor[(L.type, E.type), Float32, CPU.type] = res
      assert(res.size === Seq(L.size, E.size))
      assert(res.value === Seq(Seq(0.4962, 0.7682, 0.0884)))
    }

    it("should calculate for batched tensors") {
      case object B extends Dim.Static[4L]
      case object L extends Dim.Static[1L]
      case object S extends Dim.Static[2L]
      case object E extends Dim.Static[3L]

      val q = Tensor.rand(B, L, E)
      val k = Tensor.rand(B, S, E)
      val v = Tensor.rand(B, S, E)

      val res = functional.scaledDotProductAttention(q, k, v)
      val resType: Tensor[(B.type, L.type, E.type), Float32, CPU.type] = res
      assert(res.size === Seq(B.size, L.size, E.size))
      // No assertion on pointless result, that's covered by the previous test.
    }

  }
}
