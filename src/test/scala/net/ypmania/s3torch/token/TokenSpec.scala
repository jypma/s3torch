package net.ypmania.s3torch.token

import net.ypmania.s3torch.UnitSpec
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Device.CPU

class TokenSpec extends UnitSpec {
  describe("Token64Type") {
    case object Tok extends Token64Type

    it("can create a Tensor from a single token") {
      val t = Tensor(Tok(1))
      val tType: Tensor[EmptyTuple, Tok.DType, CPU.type] = t
      assert(t.value == Tok(1))
    }

    it("can create a Tensor from a sequence of tokens") {
      val t = Tensor(Seq(Tok(1), Tok(2), Tok(3)))
      val tType: Tensor[Tuple1[Dim.Dynamic], Tok.DType, CPU.type] = t
      assert(t.value == Seq(Tok(1), Tok(2), Tok(3)))
    }
  }

  describe("Token32Type") {
    case object Tok extends Token32Type

    it("can create a Tensor from a single token") {
      val t = Tensor(Tok(1))
      val tType: Tensor[EmptyTuple, Tok.DType, CPU.type] = t
      assert(t.value == Tok(1))
    }

    it("can create a Tensor from a sequence of tokens") {
      val t = Tensor(Seq(Tok(1), Tok(2), Tok(3)))
      val tType: Tensor[Tuple1[Dim.Dynamic], Tok.DType, CPU.type] = t
      assert(t.value == Seq(Tok(1), Tok(2), Tok(3)))
    }
  }
}
