package net.ypmania.s3torch

import net.ypmania.s3torch.Dim.*

class DimSpec extends UnitSpec {
  describe("* and /") {
    it("can multiply and divide dynamic dims with each other") {
      case class DimA(s: Long) extends Dim.Dynamic(s)
      case class DimB(s: Long) extends Dim.Dynamic(s)

      val dimA = DimA(2)
      val dimB = DimB(3)
      val dim = dimA * dimB
      val dimType: DimA * DimB = dim
      assert(dim.size == dimA.size * dimB.size)

      val div1 = dim / dimB
      val div1Type: DimA = div1
      assert(div1.size == dimA.size)

      val div2 = dim / dimA
      val div2Type: DimB = div2
      assert(div2.size == dimB.size)
    }

    it("can multiply and divide dims with a literal int") {
      case class DimA(s: Long) extends Dim.Dynamic(s)
      val dimA = DimA(2)

      val dim = dimA * 3
      val dimType: DimA * Dim.Static[3L] = dim
      assert(dim.size == dimA.size * 3)

      // TODO investigate syntax improvement for: dim / 3
      val div1 = dim / Dim.Static(3L)
      val div1Type: DimA = div1
      assert(div1.size == dimA.size)

      val div2 = dim / dimA
      val div2Type: Dim.Static[3L] = div2
      assert(div2.size == 3)
    }
  }
}
