package net.ypmania.s3torch

import org.scalatest.Assertions._

import Dim.Static
import Dim.Dynamic

class TensorSpec extends UnitSpec {
  case object ExampleStatic extends Static[10L]
  case object ExampleDynamic extends Dynamic(42)

  describe("Tensor") {
    describe("apply") {
      it("can create a Double scalar") {
        val t = Tensor(5.0)
        val tType: Tensor[EmptyTuple.type, Float32] = t
        assert(t.size == Seq[Long]())
      }
    }

    describe("zeros") {
      it("can create with dimension 1") {
        val of1static = Tensor.zeros(1L)
        val of1staticType: Tensor[Tuple1[Static[1L]], Float32] = of1static
        assert(of1static.size == Seq(1L))

        val of1named = Tensor.zeros(ExampleStatic)
        val of1namedType: Tensor[Tuple1[ExampleStatic.type], Float32] = of1named
        assert(of1named.size == Seq(10L))

        val of1dynamic = Tensor.zeros(ExampleDynamic)
        val of1dynamicType: Tensor[Tuple1[ExampleDynamic.type], Float32] = of1dynamic
        assert(of1dynamic.size == Seq(42L))
      }

      it("can create with dimension 2") {
        val of10x42 = Tensor.zeros(10L, 42L)
        val of10x42Type: Tensor[(Static[10L], Static[42L]), Float32] = of10x42
        assert(of10x42.size == Seq(10L, 42L))
      }
    }
  }
}
