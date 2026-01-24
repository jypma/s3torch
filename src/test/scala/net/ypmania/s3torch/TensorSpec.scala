package net.ypmania.s3torch

import org.scalatest.Assertions._

import Index.Slice
import Dim.Static
import Dim.Dynamic
import scala.reflect.ClassTag
import net.ypmania.s3torch.Shape.Widen
import DType.*
import Tensor.KeepDim
import net.ypmania.s3torch.Dim.*
import net.ypmania.s3torch.Shape.Select.*
import net.ypmania.s3torch.Shape.Select

class TensorSpec extends UnitSpec {
  case object ExampleStatic extends Static[10L]
  case object ExampleDynamic extends Dynamic(42)

  describe("Tensor construction") {
    describe("apply") {
      it("can create a Double scalar") {
        val t = Tensor(5.0)
        val tType: Tensor[EmptyTuple.type, Float64.type] = t
        assert(t.size == Seq[Long]())
        assert(t.value == 5.0)
      }

      it("can create an Int scalar and change defaults") {
        val t = Tensor(5, Int8)
        val tType: Tensor[EmptyTuple.type, Int8.type] = t
        assert(t.size == Seq[Long]())
        assert(t.value.isInstanceOf[Byte])
        assert(t.value == 5)
      }

      it("can create a byte vector") {
        val t = Tensor(Seq[Byte](1, 2, 3))
        val tType: Tensor[Tuple1[Dynamic], Int8.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1, 2, 3))
      }

      it("can create a dynamic double vector") {
        val t = Tensor(Seq(1.0, 2.0, 3.0))
        val tType: Tensor[Tuple1[Dynamic], Float64.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1.0, 2.0, 3.0))
      }

      it("can create a dynamic float vector") {
        // FIXME investigate what happens if we leave out [Float] here
        val t = Tensor(Seq[Float](1.0, 2.0, 3.0), Float32)
        val tType: Tensor[Tuple1[Dynamic], Float32.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1.0, 2.0, 3.0))
      }

      it("can create a static double vector") {
        val t = Tensor((1.0, 2.0, 3.0))
        val tType: Tensor[Tuple1[Static[3L]], Float64.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1.0, 2.0, 3.0))
      }

      it("can create a static byte vector") {
        val t = Tensor((1.toByte, 2.toByte, 3.toByte))
        val tType: Tensor[Tuple1[Static[3L]], Int8.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1, 2, 3))
      }

      it("can create a static matrix") {
        val t = Tensor((
          ((1,2,3)),
          ((4,5,6))
        ))
        val tType: Tensor[(Static[2L], Static[3L]), Int32.type] = t
        assert(t.size == Seq(2L, 3L))
        assert(t.value == Seq(Seq(1,2,3), Seq(4,5,6)))
      }

      it("can create a dynamic matrix") {
        val t = Tensor(Seq(
          Seq(1,2,3),
          Seq(4,5,6)
        ))
        val tType: Tensor[(Dynamic, Dynamic), Int32.type] = t
        assert(t.size == Seq(2L, 3L))
        assert(t.value == Seq(Seq(1,2,3), Seq(4,5,6)))
      }

      it("can create a mixed matrix") {
        val t = Tensor((
          Seq(1,2,3),
          Seq(4,5,6)
        ))
        val tType: Tensor[(Static[2L], Dynamic), Int32.type] = t
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
        val tType: Tensor[(Dynamic, Dynamic, Dynamic), Int32.type] = t
        assert(t.size == Seq(1L, 2L, 3L))
        assert(t.value == Seq(Seq(Seq(1,2,3), Seq(4,5,6))))
      }

      it("can create various int scalars") {
        Tensor(5, Int8)
        Tensor(5, UInt8)
        Tensor(5, Int16)
        Tensor(5, Int32)
        Tensor(5, Int64)
      }

      it("can create various float scalars") {
        Tensor(5.0, Float16)
        Tensor(5.0, Float32)
        Tensor(5.0, Float64)
      }
    }

    describe("arange") {
      it("can create a range from ints") {
        val t = Tensor.arange(0, 3, 1)
        val tType: Tensor[Tuple1[Dynamic], Int32.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(0, 1, 2))
      }

      it("can create a range from doubles") {
        val t = Tensor.arange(0.0, 3.0, 1.0)
        val tType: Tensor[Tuple1[Dynamic], Float64.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(0, 1, 2))
      }

      it("can create a range from Dim") {
        val t = Tensor.arangeOf(ExampleStatic)
        val tType: Tensor[Tuple1[ExampleStatic.type], Int64.type] = t
        assert(t.size == Seq(10L))
        assert(t.value.toSeq == Seq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
      }

      it("can create a range from unknown Dim") {
        val dim: Dim = ExampleStatic
        val t = Tensor.arangeOf(dim)
        val tType: Tensor[Tuple1[Dim], Int64.type] = t
        assert(t.size == Seq(10L))
        assert(t.value.toSeq == Seq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
      }

      it("can create a float range from Dim") {
        val t = Tensor.arangeOf(ExampleStatic, Float32)
        val tType: Tensor[Tuple1[ExampleStatic.type], Float32.type] = t
        assert(t.size == Seq(10L))
        assert(t.value.toSeq == Seq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
      }
    }

    describe("rand") {
      it("can generate random numbers using fixed seed") {
        // Seed provided by given RandomSource in UnitTest.scala
        val t = Tensor.rand(3L)
        assert(t.size == Seq(3L))
        assert(t.value.toSeq === Seq(0.4962, 0.7682, 0.0884))
      }
    }

    describe("zeros") {
      it("can create with dimension 1") {
        val of1static = Tensor.zeros(1L)
        val of1staticType: Tensor[Tuple1[Static[1L]], Float32.type] = of1static
        assert(of1static.size == Seq(1L))

        val of1named = Tensor.zeros(ExampleStatic)
        val of1namedType: Tensor[Tuple1[ExampleStatic.type], Float32.type] = of1named
        assert(of1named.size == Seq(10L))

        val of1dynamic = Tensor.zeros(ExampleDynamic)
        val of1dynamicType: Tensor[Tuple1[ExampleDynamic.type], Float32.type] = of1dynamic
        assert(of1dynamic.size == Seq(42L))
      }

      it("can create with dimension 2") {
        val of10x42 = Tensor.zeros(10L, 42L)
        val of10x42Type: Tensor[(Static[10L], Static[42L]), Float32.type] = of10x42
        assert(of10x42.size == Seq(10L, 42L))
      }
    }
  }

  describe("Tensor") {
    describe("flatten") {
      it("can flatten a 1D tensor") {
        val t = Tensor((1, 2, 3))
        val r = t.flatten
        val rType: Tensor[Tuple1[Static[3L]], Int32.type] = r
        assert(r.size == Seq(3L))
        assert(r.value.toSeq == Seq(1, 2, 3))
      }
    }

    describe("mean") {
      case object DimA extends Dim.Static[2L]
      case object DimB extends Dim.Static[3L]

      it("can calculate mean of first dim") {
        var t = Tensor.zeros(DimA, DimB)
        t((0,0)) = 3.0
        t((1,0)) = 2.0
        val res = t.meanBy(DimA)
        val resType: Tensor[DimB.type *: EmptyTuple, Float32.type] = res
        assert(res.size == Seq(3L))
        assert(res.value.toSeq == Seq(2.5, 0, 0))
      }

      it("can calculate mean of second dim") {
        var t = Tensor.zeros(DimA, DimB)
        t((0,0)) = 3.0
        t((1,0)) = 2.0
        val res = t.meanBy(DimB)
        val resType: Tensor[DimA.type *: EmptyTuple, Float32.type] = res
        assert(res.size == Seq(2L))
        assert(res.value.toSeq === Seq(1.0, 0.6666))
      }

      it("can calculate mean of selected dim and keep it") {
        var t = Tensor.zeros(DimA, DimB)
        t((0,0)) = 3.0
        t((1,0)) = 2.0
        val res = t.meanBy(DimA)(using KeepDim)
        val resType: Tensor[(Dim.One, DimB.type), Float32.type] = res
        assert(res.size == Seq(1L, 3L))
        assert(res.value.toSeq == Seq(Seq(2.5, 0, 0)))
      }
    }

    describe("plus") {
      it("can add a primitive") {
        val t = Tensor((1, 2, 3))
        val r = t + 1
        val rType: Tensor[Tuple1[Static[3L]], Int32.type] = r
        assert(r.size == Seq(3L))
        assert(r.value.toSeq == Seq(2, 3, 4))
      }

      it("can add vector and scalar") {
        val a = Tensor((1, 2, 3))
        val b = Tensor(1)
        val r = a + b
        val rType: Tensor[Tuple1[Static[3L]], Int32.type] = r
        assert(r.size == Seq(3L))
        assert(r.value.toSeq == Seq(2, 3, 4))
      }

      it("can add vectors of different lengths") {
        val a = Tensor((1, 2, 3))
        val b = Tensor(Tuple1(1))
        val r = a + b
        val rType: Tensor[Tuple1[Static[3L]], Int32.type] = r
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
        val rType: Tensor[(Static[4L], Static[4L]), Int32.type] = r
        assert(r.size == Seq(4L, 4L))
        assert(r.value.toSeq == Seq(
          Seq(6, 7, 8, 9),
          Seq(7, 8, 9, 10),
          Seq(8, 9, 10, 11),
          Seq(9, 10, 11, 12))
        )
      }
    }

    describe("std") {
      it("can calculate standard deviation") {
        var t = Tensor((1.0, 2.0, 3.0))
        val res = t.stdBy(Shape.Select.First)
        val resType: Tensor[EmptyTuple, Float64.type] = res
        assert(res.size == Seq())
        assert(res.value == 1.0)
      }
    }

    describe("transpose") {
      it("can swap two dims of a 3-dim tensor") {
        val a = Tensor((
          ((
            ((1,2,3)),
            ((4,5,6))
          )),
          ((
            ((7,8,9)),
            ((10,11,12))
          ))
        ))
        val aType: Tensor[(Static[2L], Static[2L], Static[3L]), Int32.type] = a
        val b = a.transpose(Shape.Select.Idx(0), Shape.Select.Idx(2))
        val bType: Tensor[(Static[3L], Static[2L], Static[2L]), Int32.type] = b
        assert(b.value == Seq(
          Seq(
            Seq(1, 7),
            Seq(4, 10)
          ), Seq(
            Seq(2, 8),
            Seq(5, 11)
          ), Seq(
            Seq(3, 9),
            Seq(6, 12)
          )
        ))
      }

      it("can transpose a matrix without extra args") {
        val m = Tensor(
          Tuple1(
            ((1, 2))
          )
        )
        val r = m.t
        val rType: Tensor[(Static[2L], Static[1L]), Int32.type] = r
        assert(r.value === Seq(
          Seq(1),
          Seq(2)
        ))
      }
    }

    describe("update") {
      it("can set a single value in a vector") {
        val a = Tensor((1, 2, 3))
        a(0) = 4
        assert(a.value.toSeq == Seq(4,2,3))
      }

      it("can set a scalar") {
        val a = Tensor(1.0)
        a(EmptyTuple) = 4
        assert(a.value == 4)
      }

      it("can set a value by specifying the dimension") {
        case object MyDim extends Dim.Static[3L]
        val a = Tensor.zeros(MyDim)
        a(MyDim -> 0) = 4
        assert(a.value.toSeq == Seq(4,0,0))
      }

      it("can set a single value in a matrix") {
        val t = Tensor((
          ((1,2,3)),
          ((4,5,6))
        ))
        t((1, 1)) = 9
        assert(t.value.toSeq == Seq(Seq(1,2,3), Seq(4,9,6)))
      }

      it("can set a value in a matrix by specifying the dimension, as tuple or args") {
        case object MyDimA extends Dim.Static[3L]
        case object MyDimB extends Dim.Static[2L]
        val a = Tensor.zeros(MyDimA, MyDimB)
        // If these are swapped, we get a nice compile error.
        a((MyDimA -> 0, MyDimB -> 1)) = 9
        assert(a.value.toSeq == Seq(Seq(0,9),Seq(0,0),Seq(0,0)))
      }

      it("can set a slice of a vector") {
        val v = Tensor.zeros(6L)
        var ones = Tensor((1, 1, 1))
        v(Slice(step = 2)) = ones
        assert(v.value.toSeq == Seq(1, 0, 1, 0, 1, 0))
      }
    }

    describe("unsqueeze") {
      case object DimA extends Static[2L]
      case object DimB extends Static[3L]
      val vector = Tensor.zeros(DimA)
      val matrix = Tensor.zeros(DimA, DimB)

      it("can unsqueeze after last") {
        val r = vector.unsqueezeAfter(Shape.Select.Last)
        val rType: Tensor[(DimA.type, Static[1L]), Float32.type] = r
        assert(r.size == Seq(2L, 1L))
        assert(r.value.toSeq == Seq(Seq(0), Seq(0)))
      }

      it("can unsqueeze after the last dim of a matrix") {
        // Verify that we can unsqueeze by type as well
        val r2 = matrix.unsqueezeAfter[DimB.type]
        val r2Type: Tensor[(DimA.type, DimB.type, Static[1L]), Float32.type] = r2

        val r = matrix.unsqueezeAfter(DimB)
        val rType: Tensor[(DimA.type, DimB.type, Static[1L]), Float32.type] = r
        assert(r.size == Seq(2L, 3L, 1L))
        assert(r.value.toSeq == Seq(
          Seq(Seq(0),Seq(0),Seq(0)),
          Seq(Seq(0),Seq(0),Seq(0))
        ))
      }

      it("can unsqueeze after the first dim of a matrix") {
        val r = matrix.unsqueezeAfter(DimA)
        val rType: Tensor[(DimA.type, Static[1L], DimB.type), Float32.type] = r
        assert(r.size == Seq(2L, 1L, 3L))
        assert(r.value.toSeq == Seq(
          Seq(
            Seq(0,0,0)
          ),
          Seq(
            Seq(0,0,0)
          ),
        ))
      }

      it("can unsqueeze before first") {
        val r = vector.unsqueezeBefore(Shape.Select.First)
        val rType: Tensor[(Static[1L], DimA.type), Float32.type] = r
        assert(r.size == Seq(1L, 2L))
        assert(r.value.toSeq == Seq(Seq(0, 0)))
      }

      it("can unsqueeze before first (by index)") {
        val r = vector.unsqueezeBefore(Shape.Select.Idx(0))
        val rType: Tensor[(Static[1L], DimA.type), Float32.type] = r
        assert(r.size == Seq(1L, 2L))
        assert(r.value.toSeq == Seq(Seq(0, 0)))
      }

      it("can unsequeeze before first dim of a matrix") {
        val r = matrix.unsqueezeBefore(DimA)
        val rType: Tensor[(Static[1L], DimA.type, DimB.type), Float32.type] = r
        assert(r.size == Seq(1L, 2L, 3L))
        assert(r.value.toSeq == Seq(
          Seq(
            Seq(0,0,0),
            Seq(0,0,0)
          )
        ))
      }

      it("can unsequeeze before last dim of a matrix") {
        val r = matrix.unsqueezeBefore(DimB)
        val rType: Tensor[(DimA.type, Static[1L], DimB.type), Float32.type] = r
        assert(r.size == Seq(2L, 1L, 3L))
        assert(r.value.toSeq == Seq(
          Seq(
            Seq(0,0,0)
          ),
          Seq(
            Seq(0,0,0)
          ),
        ))
      }
    }

    describe("split and unsplit") {
      case object DimA extends Static[6L]
      case object DimB extends Static[3L]

      it("can split on specific dim") {
        val matrix = Tensor.zeros(DimA, DimB)
        matrix((1, 1)) = 1.0
        val res = matrix.split(DimA).into[2L]
        val resType: Tensor[(Static[2L], DimA.type / 2L, DimB.type), Float32.type] = res
        assert(res.size == Seq(2L, 3L, 3L))
        assert(res.value(0)(1)(1) == 1.0)

        val un = res.unsplit(Divided(DimA))
        assert(un.size == Seq(6L, 3L))
        assert(un.value(1)(1) == 1.0)
      }

      it("can split on last") {
        case object DimC extends Static[4L]
        val t = Tensor.zeros(DimA, DimB, DimC)
        val res = t.split(DimC).into[4L]
        val resType: Tensor[(DimA.type, DimB.type, Static[4L], DimC.type / 4L), Float32.type] = res
        assert(res.size == Seq(6L, 3L, 4L, 1L))

        val un = res.unsplit(Divided(DimC))
        assert(un.size == Seq(6L, 3L, 4L))
      }
    }
  }
}
