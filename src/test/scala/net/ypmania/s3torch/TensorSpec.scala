package net.ypmania.s3torch

import org.scalatest.Assertions._

import Dim.Static
import Dim.Dynamic
import scala.reflect.ClassTag
import net.ypmania.s3torch.internal.Broadcast.MaxEachDim
import net.ypmania.s3torch.Shape.Widen

class TensorSpec extends UnitSpec {
  case object ExampleStatic extends Static[10L]
  case object ExampleDynamic extends Dynamic(42)

  describe("Tensor construction") {
    describe("apply") {
      it("can create a Double scalar") {
        val t = Tensor(5.0)
        val tType: Tensor[EmptyTuple.type, Float64] = t
        assert(t.size == Seq[Long]())
        assert(t.value == 5.0)
      }

      it("can create an Int scalar and change defaults") {
        val t = Tensor(5, int8)
        val tType: Tensor[EmptyTuple.type, Int8] = t
        assert(t.size == Seq[Long]())
        assert(t.value.isInstanceOf[Byte])
        assert(t.value == 5)
      }

      it("can create a byte vector") {
        val t = Tensor(Seq[Byte](1, 2, 3))
        val tType: Tensor[Tuple1[Dynamic], Int8] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1, 2, 3))
      }

      it("can create a dynamic double vector") {
        val t = Tensor(Seq(1.0, 2.0, 3.0))
        val tType: Tensor[Tuple1[Dynamic], Float64] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1.0, 2.0, 3.0))
      }

      it("can create a dynamic float vector") {
        // FIXME investigate what happens if we leave out [Float] here
        val t = Tensor(Seq[Float](1.0, 2.0, 3.0), float32)
        val tType: Tensor[Tuple1[Dynamic], Float32] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1.0, 2.0, 3.0))
      }

      it("can create a static double vector") {
        val t = Tensor((1.0, 2.0, 3.0))
        val tType: Tensor[Tuple1[Static[3L]], Float64] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1.0, 2.0, 3.0))
      }

      it("can create a static byte vector") {
        val t = Tensor((1.toByte, 2.toByte, 3.toByte))
        val tType: Tensor[Tuple1[Static[3L]], Int8] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1, 2, 3))
      }

      it("can create a static matrix") {
        val t = Tensor((
          ((1,2,3)),
          ((4,5,6))
        ))
        val tType: Tensor[(Static[2L], Static[3L]), Int32] = t
        assert(t.size == Seq(2L, 3L))
        assert(t.value == Seq(Seq(1,2,3), Seq(4,5,6)))
      }

      it("can create a dynamic matrix") {
        val t = Tensor(Seq(
          Seq(1,2,3),
          Seq(4,5,6)
        ))
        val tType: Tensor[(Dynamic, Dynamic), Int32] = t
        assert(t.size == Seq(2L, 3L))
        assert(t.value == Seq(Seq(1,2,3), Seq(4,5,6)))
      }

      it("can create a mixed matrix") {
        val t = Tensor((
          Seq(1,2,3),
          Seq(4,5,6)
        ))
        val tType: Tensor[(Static[2L], Dynamic), Int32] = t
        assert(t.size == Seq(2L, 3L))
        assert(t.value == Seq(Seq(1,2,3), Seq(4,5,6)))
      }

      it("can create a 3D tensor") {
        val t = Tensor(Seq(
          Seq(
            Seq(1,2,3),
            Seq(4,5,6)
          )
        ))
        val tType: Tensor[(Dynamic, Dynamic, Dynamic), Int32] = t
        assert(t.size == Seq(1L, 2L, 3L))
        assert(t.value == Seq(Seq(Seq(1,2,3), Seq(4,5,6))))
      }

      it("can create various int scalars") {
        Tensor(5, int8)
        Tensor(5, uint8)
        Tensor(5, int16)
        Tensor(5, int32)
        Tensor(5, int64)
      }

      it("can create various float scalars") {
        Tensor(5.0, float16)
        Tensor(5.0, float32)
        Tensor(5.0, float64)
      }
    }

    describe("arange") {
      it("can create a range from ints") {
        val t = Tensor.arange(0, 3, 1)
        val tType: Tensor[Tuple1[Dynamic], Int32] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(0, 1, 2))
      }

      it("can create a range from doubles") {
        val t = Tensor.arange(0.0, 3.0, 1.0)
        val tType: Tensor[Tuple1[Dynamic], Float64] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(0, 1, 2))
      }

      it("can create a range from Dim") {
        val t = Tensor.arangeOf(ExampleStatic)
        val tType: Tensor[Tuple1[ExampleStatic.type], Int64] = t
        assert(t.size == Seq(10L))
        assert(t.value.toSeq == Seq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
      }

      it("can create a float range from Dim") {
        val t = Tensor.arangeOf(ExampleStatic, float32)
        val tType: Tensor[Tuple1[ExampleStatic.type], Float32] = t
        assert(t.size == Seq(10L))
        assert(t.value.toSeq == Seq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
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

  describe("Tensor") {
    describe("flatten") {
      it("can flatten a 1D tensor") {
        val t = Tensor((1, 2, 3))
        val r = t.flatten
        val rType: Tensor[Tuple1[Static[3L]], Int32] = r
        assert(r.size == Seq(3L))
        assert(r.value.toSeq == Seq(1, 2, 3))
      }
    }

    describe("plus") {
      it("can add a primitive") {
        val t = Tensor((1, 2, 3))
        val r = t + 1
        val rType: Tensor[Tuple1[Static[3L]], Int32] = r
        assert(r.size == Seq(3L))
        assert(r.value.toSeq == Seq(2, 3, 4))
      }

      it("can add vector and scalar") {
        val a = Tensor((1, 2, 3))
        val b = Tensor(1)
        val r = a + b
        val rType: Tensor[Tuple1[Static[3L]], Int32] = r
        assert(r.size == Seq(3L))
        assert(r.value.toSeq == Seq(2, 3, 4))
      }

      it("can add vectors of different lengths") {
        val a = Tensor((1, 2, 3))
        val b = Tensor(Tuple1(1))
        val r = a + b
        val rType: Tensor[Tuple1[Static[3L]], Int32] = r
        assert(r.size == Seq(3L))
        assert(r.value.toSeq == Seq(2, 3, 4))
      }

      it("can add a vector to a matrix") {
        val a = Tensor((1, 2, 3, 4)) // [4]
        val b = Tensor((             // [4, 1]
          Tuple1(5),
          Tuple1(6),
          Tuple1(7),
          Tuple1(8)
        ))
        val r = a + b
        val rType: Tensor[(Static[4L], Static[4L]), Int32] = r
        assert(r.size == Seq(4L, 4L))
        assert(r.value.toSeq == Seq(
          Seq(6, 7, 8, 9),
          Seq(7, 8, 9, 10),
          Seq(8, 9, 10, 11),
          Seq(9, 10, 11, 12))
        )
      }
    }

    describe("unsqueeze") {
      case object DimA extends Static[2L]
      case object DimB extends Static[3L]
      val vector = Tensor.zeros(DimA)
      val matrix = Tensor.zeros(DimA, DimB)

      it("can unsqueeze after last") {
        val r = vector.unsqueezeAfterLast
        val rType: Tensor[(DimA.type, Static[1L]), Float32] = r
        assert(r.size == Seq(2L, 1L))
        assert(r.value.toSeq == Seq(Seq(0), Seq(0)))
      }

      it("can unsqueeze after the last dim of a matrix") {
        val r = matrix.unsqueezeAfter(DimB)
        val rType: Tensor[(DimA.type, Static[1L], DimB.type), Float32] = r
        assert(r.size == Seq(2L, 1L, 3L))
        assert(r.value.toSeq == Seq(
          Seq(Seq(0,0,0)),
          Seq(Seq(0,0,0))
        ))
      }

      it("can unsqueeze after the first dim of a matrix") {
        val r = matrix.unsqueezeAfter(DimA)
        val rType: Tensor[(Static[1L], DimA.type, DimB.type), Float32] = r
        assert(r.size == Seq(1L, 2L, 3L))
        assert(r.value.toSeq == Seq(
          Seq(
            Seq(0,0,0),
            Seq(0,0,0)
          ),
        ))
      }
    }
  }
}
