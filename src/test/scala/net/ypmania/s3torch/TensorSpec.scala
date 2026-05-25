package net.ypmania.s3torch

import Device.CUDA
import Index.Slice
import Index.given
import Dim.Static
import Dim.Dynamic
import scala.reflect.ClassTag
import DType.*
import net.ypmania.s3torch.Dim.*
import net.ypmania.s3torch.Select
import net.ypmania.s3torch.Select.First
import net.ypmania.s3torch.Select.Last
import net.ypmania.s3torch.Select.dim
import net.ypmania.s3torch.Shape.Scalar
import net.ypmania.s3torch.internal.Broadcast
import internal.MatMul
import Device.CPU
import Tuple.:*

class TensorSpec extends UnitSpec {
  case object ExampleStatic extends Static[10L]
  case object ExampleDynamic extends Dynamic(42)

  describe("Tensor construction") {
    describe("apply") {
      it("can create a Double scalar") {
        val t = Tensor(5.0)
        val tType: Tensor[EmptyTuple.type, Float64, CPU.type] = t
        assert(t.size == Seq[Long]())
        assert(t.value == 5.0)
      }

      it("can create an Int scalar and change defaults") {
        val t = Tensor(5.toByte).to(int8)
        val tType: Tensor[EmptyTuple.type, Int8, CPU.type] = t
        assert(t.size == Seq[Long]())
        assert(t.value.isInstanceOf[Byte])
        assert(t.value == 5)
      }

      it("can create a byte vector") {
        val t = Tensor(Seq[Byte](1, 2, 3))
        val tType: Tensor[Tuple1[Dynamic], Int8, CPU.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1, 2, 3))
      }

      it("can create a dynamic double vector") {
        val t = Tensor(Seq(1.0, 2.0, 3.0))
        val tType: Tensor[Tuple1[Dynamic], Float64, CPU.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1.0, 2.0, 3.0))
      }

      it("can create a dynamic float vector") {
        val t = Tensor(Seq(1.0, 2.0, 3.0)).to(float32)
        val tType: Tensor[Tuple1[Dynamic], Float32, CPU.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1.0, 2.0, 3.0))
      }

      it("can create a static double vector") {
        val t = Tensor((1.0, 2.0, 3.0))
        val tType: Tensor[Tuple1[Static[3L]], Float64, CPU.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1.0, 2.0, 3.0))
      }

      it("can create a static byte vector") {
        val t = Tensor((1.toByte, 2.toByte, 3.toByte))
        val tType: Tensor[Tuple1[Static[3L]], Int8, CPU.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(1, 2, 3))
      }

      it("can create a static boolean vector") {
        val t = Tensor((true, true, false))
        val tType: Tensor[Tuple1[Static[3L]], Bool, CPU.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(true, true, false))
      }

      it("can create a static matrix") {
        val t = Tensor((
          ((1,2,3)),
          ((4,5,6))
        ))
        val tType: Tensor[(Static[2L], Static[3L]), Int32, CPU.type] = t
        assert(t.size == Seq(2L, 3L))
        assert(t.value == Seq(Seq(1,2,3), Seq(4,5,6)))
      }

      it("can create a static byte matrix") {
        val t = Tensor((
          ((1,2,3)),
          ((4,5,6))
        )).to(int8)
        val tType: Tensor[(Static[2L], Static[3L]), Int8, CPU.type] = t
        assert(t.size == Seq(2L, 3L))
        assert(t.value == Seq(Seq(1,2,3), Seq(4,5,6)))
      }

      it("can create a dynamic matrix") {
        val t = Tensor(Seq(
          Seq(1,2,3),
          Seq(4,5,6)
        ))
        val tType: Tensor[(Dynamic, Dynamic), Int32, CPU.type] = t
        assert(t.size == Seq(2L, 3L))
        assert(t.value == Seq(Seq(1,2,3), Seq(4,5,6)))
      }

      it("can create a mixed matrix") {
        val t = Tensor((
          Seq(1,2,3),
          Seq(4,5,6)
        ))
        val tType: Tensor[(Static[2L], Dynamic), Int32, CPU.type] = t
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
        val tType: Tensor[(Dynamic, Dynamic, Dynamic), Int32, CPU.type] = t
        assert(t.size == Seq(1L, 2L, 3L))
        assert(t.value == Seq(Seq(Seq(1,2,3), Seq(4,5,6))))
      }

      it("can create various int scalars") {
        Tensor(5).to(int8)
        Tensor(5).to(uint8)
        Tensor(5).to(int16)
        Tensor(5).to(int32)
        Tensor(5).to(int64)
      }

      it("can create various float scalars") {
        Tensor(5.0).to(float16)
        Tensor(5.0).to(float32)
        Tensor(5.0).to(float64)
      }

      it("can create, modify and read back a Float16 vector") {
        val t = Tensor((1.0, 2.0, 3.0)).to(float16)
        val tType: Tensor[Static[3L] *: EmptyTuple, Float16, CPU.type] = t
        assert(t(0).value === 1.0)
        t(0) = 4.0
        val r = t.value
        assert(r.toSeq === Seq(4.0, 2.0, 3.0))
      }

      it("can create, modify and read back a BFloat16 vector") {
        val t = Tensor((1.0, 2.0, 3.0)).to(bfloat16)
        val tType: Tensor[Static[3L] *: EmptyTuple, BFloat16, CPU.type] = t
        assert(t(0).value === 1.0)
        t(0) = 4.0
        val r = t.value
        assert(r.toSeq === Seq(4.0, 2.0, 3.0))
      }
    }

    describe("arange") {
      it("can create a range from ints") {
        val t = Tensor.arange(0, 3, 1)
        val tType: Tensor[Tuple1[Dynamic], Int32, CPU.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(0, 1, 2))
      }

      it("can create a range from doubles") {
        val t = Tensor.arange(0.0, 3.0, 1.0)
        val tType: Tensor[Tuple1[Dynamic], Float64, CPU.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq == Seq(0, 1, 2))
      }

      it("can create a range from Dim, in natural DType") {
        val t = Tensor.arangeOf(ExampleStatic)
        val tType: Tensor[Tuple1[ExampleStatic.type], Int64, CPU.type] = t
        assert(t.size == Seq(10L))
        assert(t.value.toSeq == Seq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
      }

      it("can create a range from Dim, in default DType") {
        val t = Tensor.arangeOfD(ExampleStatic)
        val tType: Tensor[Tuple1[ExampleStatic.type], Float32, CPU.type] = t
        assert(t.size == Seq(10L))
        assert(t.value.toSeq == Seq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
      }

      it("can create a range from unknown Dim") {
        val dim: Dim = ExampleStatic
        val t = Tensor.arangeOfD(dim)
        val tType: Tensor[Tuple1[Dim], Float32, CPU.type] = t
        assert(t.size == Seq(10L))
        assert(t.value.toSeq == Seq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
      }

      it("can create a range from a Dim.Ref") {
        val dim = Dim.Ref(ExampleStatic)
        val t = Tensor.arangeOfD(dim)
        val tType: Tensor[Tuple1[ExampleStatic.type], Float32, CPU.type] = t
      }
    }

    describe("full") {
      it("can fill a vector with a value, with natural DType") {
        val t = Tensor.full(5)(ExampleStatic)
        val tType: Tensor[Tuple1[ExampleStatic.type], Int32, CPU.type] = t
        assert(t.size == Seq(10L))
        assert(t.value.toSeq == Seq(5, 5, 5, 5, 5, 5, 5, 5, 5, 5))
      }

      it("can fill a vector with a value, with default DType") {
        val t = Tensor.fullD(5.0)(ExampleStatic)
        val tType: Tensor[Tuple1[ExampleStatic.type], Float32, CPU.type] = t
        assert(t.size == Seq(10L))
        assert(t.value.toSeq == Seq(5, 5, 5, 5, 5, 5, 5, 5, 5, 5))
      }

    }

    describe("stack") {
      it("can combine two tensors into a type-only batch") {
        val a = Tensor((1, 2, 3))
        val b = Tensor((4, 5, 6))
        trait Batch extends Dim
        val c = Tensor.stack[Batch](Seq(a, b))
        val cType: Tensor[(Batch, Static[3L]), Int32, CPU.type] = c
        assert(c.size == Seq(2L, 3L))
        assert(c.value.toSeq == Seq(
          Seq(1, 2, 3),
          Seq(4, 5, 6)
        ))
      }

      it("can combine two tensors into a value-based batch") {
        val a = Tensor((1, 2, 3))
        val b = Tensor((4, 5, 6))
        case object Batch extends Dim.Static[2L]
        val c = Tensor.stack(Batch)(Seq(a, b))
        val cType: Tensor[(Batch.type, Static[3L]), Int32, CPU.type] = c
        assert(c.size == Seq(2L, 3L))
        assert(c.value.toSeq == Seq(
          Seq(1, 2, 3),
          Seq(4, 5, 6)
        ))
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

    describe("randint") {
      it("can generate random numbers using fixed seed") {
        // Seed provided by given RandomSource in UnitTest.scala
        val t = Tensor.randint(42)(3L)
        val tType: Tensor[Tuple1[Static[3L]], Int64, CPU.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq === Seq(32, 3, 35))
      }
    }

    describe("randintD") {
      it("can generate random numbers using fixed seed") {
        // Seed provided by given RandomSource in UnitTest.scala
        val t = Tensor.randintD(42)(3L)
        val tType: Tensor[Tuple1[Static[3L]], Float32, CPU.type] = t
        assert(t.size == Seq(3L))
        assert(t.value.toSeq === Seq(32.0, 3.0, 35.0))
      }
    }

    describe("zeros") {
      it("can create with dimension 1") {
        val of1static = Tensor.zeros(1L)
        val of1staticType: Tensor[Tuple1[Static[1L]], Float32, CPU.type] = of1static
        assert(of1static.size == Seq(1L))

        val of1named = Tensor.zeros(ExampleStatic)
        val of1namedType: Tensor[Tuple1[ExampleStatic.type], Float32, CPU.type] = of1named
        assert(of1named.size == Seq(10L))

        val of1dynamic = Tensor.zeros(ExampleDynamic)
        val of1dynamicType: Tensor[Tuple1[ExampleDynamic.type], Float32, CPU.type] = of1dynamic
        assert(of1dynamic.size == Seq(42L))
      }

      it("can create with dimension 2") {
        val of10x42 = Tensor.zeros(10L, 42L)
        val of10x42Type: Tensor[(Static[10L], Static[42L]), Float32, CPU.type] = of10x42
        assert(of10x42.size == Seq(10L, 42L))
      }
    }
  }

  describe("Tensor") {
    describe("#==") {
      it("can compare two tensors") {
        val a = Tensor((1, 2, 3))
        val b = Tensor((1, 42, 3))
        val res = a #== b
        val resType: Tensor[Tuple1[Static[3L]], Bool, CPU.type] = res
        assert(res.size == Seq(3L))
        assert(res.value.toSeq == Seq(true, false, true))
      }

      it("can compare tensor with a number") {
        val a = Tensor((
          ((1, 2)),
          ((3, 4))
        ))
        val res = a #== 1
        val resType: Tensor[(Static[2L], Static[2L]), Bool, CPU.type] = res
        assert(res.size == Seq(2L, 2L))
        assert(res.value.toSeq == Seq(Seq(true, false), Seq(false, false)))
      }

      it("can compare two tensors of different type") {
        val a = Tensor((1, 2, 3))
        val b = Tensor((1.0, 42.0, 3.0))
        val res = a #== b
        val resType: Tensor[Tuple1[Static[3L]], Bool, CPU.type] = res
        assert(res.size == Seq(3L))
        assert(res.value.toSeq == Seq(true, false, true))
      }

      it("can compare tensor with a batch") {
        val a = Tensor((1, 2, 3))
        val b = Tensor((
          ((1, 2, 3)),
          ((0, 2, 0))
        ))
        val res = a #== b
        val resType: Tensor[(Static[2L], Static[3L]), Bool, CPU.type] = res
        assert(res.size == Seq(2L, 3L))
        assert(res.value.toSeq == Seq(
          Seq(true, true, true),
          Seq(false, true, false)
        ))
      }

      it("can compare a batch with a tensor") {
        val a = Tensor((
          ((1, 2, 3)),
          ((0, 2, 0))
        ))
        val b = Tensor((1, 2, 3))
        val res = a #== b
        val resType: Tensor[(Static[2L], Static[3L]), Bool, CPU.type] = res
        assert(res.size == Seq(2L, 3L))
        assert(res.value.toSeq == Seq(
          Seq(true, true, true),
          Seq(false, true, false)
        ))
      }
    }

    describe("apply") {
      val v = Tensor((1, 2, 3))
      trait Row extends Dim.Dynamic
      trait Column extends Dim.Dynamic
      trait ZPos extends Dim.Dynamic
      val m = Tensor((
        ((1, 2, 3)),
        ((0, 2, 0))
      )).shaped[(Row, Column)]
      val cube = Tensor((
        ((
          ((1, 2, 3)),
          ((4, 5, 6))
        )),
        ((
          ((7, 8, 9)),
          ((10, 11, 12))
        ))
      )).shaped[(ZPos, Row, Column)]
      it("can select one element from a static vector statically") {
        // v(3) will nicely give a compile error here.
        val r = v(0)
        assert(r.size == Seq())
        assert(r.value == 1)
      }

      it("can select one element from a static vector dynamically") {
        val index = 1 - 1
        val r = v(index)
        val rType: Tensor[EmptyTuple, Int32, CPU.type] = r
        assert(r.size == Seq())
        assert(r.value == 1)
      }

      it("can select one element from a dynamically-sized vector") {
        val dv = Tensor(Seq(1,2,3,4,5))
        val r = dv(0)
        val rType: Tensor[EmptyTuple, Int32, CPU.type] = r
        assert(r.size == Seq())
        assert(r.value == 1)
      }

      it("can select the whole vector") {
        val r = v(Index.All)
        val rType: Tensor[Static[3L] *: EmptyTuple, Int32, CPU.type] = r
        assert(r.size == Seq(3L))
      }

      it("can select one row from a matrix using explicit dimensions") {
        val r = m(dim[Row] % 0)
        val rType: Tensor[Tuple1[Column], Int32, CPU.type] = r
        assert(r.size == Seq(3L))
        assert(r.value === Seq(1, 2, 3))
      }

      it("can select one column from a matrix using explcit dimensions") {
        val r = m(dim[Column] % 0)
        val rType: Tensor[Tuple1[Row], Int32, CPU.type] = r
        assert(r.size == Seq(2L))
        assert(r.value === Seq(1, 0))
      }

      it("can select one element from a matrix using explicit dimensions") {
        val r = m(
          dim[Row] % 0,
          dim[Column] % 1
        )
        val rType: Tensor[EmptyTuple, Int32, CPU.type] = r
        assert(r.size == Seq())
        assert(r.value == 2)
      }

      it("can select one element from a matrix using explicit dimensions and Index subclasses") {
        val r = m(
          dim[Row] % Index.First,
          dim[Column] % Index.Last
        )
        val rType: Tensor[EmptyTuple, Int32, CPU.type] = r
        assert(r.size == Seq())
        assert(r.value == 3)
      }

      it("can select one element from a cube using explicit dimensions and Index subclasses") {
        val r = cube(
          dim[ZPos] % Index.First,
          dim[Row] % Index.First,
          dim[Column] % Index.Last,
        )
        val rType: Tensor[EmptyTuple, Int32, CPU.type] = r
        assert(r.size == Seq())
        assert(r.value == 3)
      }

      it("can select one vector from a cube using 2 explicit dimensions") {
        val r = cube(
          dim[ZPos] % Index.First,
          dim[Column] % Index.Last,
        )
        val rType: Tensor[Row *: EmptyTuple.type, Int32, CPU.type] = r
        assert(r.size == Seq(2))
        assert(r.value === Seq(3, 6))
      }

      it("can select one element from a matrix using positions") {
        val r = m(0, 1)
        val rType: Tensor[EmptyTuple, Int32, CPU.type] = r
        assert(r.size == Seq())
        assert(r.value == 2)
      }

      it("can select one column from a matrix") {
        val r = m(0, Index.All)
        val rType: Tensor[Tuple1[Column], Int32, CPU.type] = r
        assert(r.size == Seq(3L))
        assert(r.value === Seq(1, 2, 3))
      }

      it("can select one row from a matrix") {
        val r = m(Index.All, 0)
        val rType: Tensor[Tuple1[Row], Int32, CPU.type] = r
        assert(r.size == Seq(2L))
        assert(r.value === Seq(1, 0))
      }

      it("can reduce a dimension using Take") {
        case object D2 extends Dim.Static[2L]
        val r = v(Index.Take(D2))
        val rType: Tensor[Tuple1[D2.type], Int32, CPU.type] = r
        assert(r.size == Seq(2L))
        assert(r.value === Seq(1, 2))
      }

      it("can reduce a dimension using Take with drop") {
        case object D2 extends Dim.Static[2L]
        val r = v(Index.Take(D2, drop = 1))
        val rType: Tensor[Tuple1[D2.type], Int32, CPU.type] = r
        assert(r.size == Seq(2L))
        assert(r.value === Seq(2, 3))
      }

      it("can reduce a dimension to the Dim of another tensor") {
        case object D2 extends Dim.Static[2L]
        val t2 = Tensor.zeros(D2)
        val r = v(Index.Take(t2.sizeOf(dim[D2.type])))
        val rType: Tensor[Tuple1[D2.type], Int32, CPU.type] = r
        assert(r.size == Seq(2L))
        assert(r.value === Seq(1, 2))
      }
    }

    describe("1-arg apply, when used on a batched one-dim tensor") {
      def doIt[B <: Shape, L1 <: Dim, S <: Shape, T <: DType](t: Tensor[S, T, CPU.type])(using Batched1[B, L1, S]): Tensor[B, T, CPU.type] = {
        val r = t(dim[L1] % Index.Last)
        val rType: Tensor[B, T, CPU.type] = r
        r
      }

      it("should accept a vector") {
        case object DimA extends Dim.Static[3L]
        val m1 = Tensor.arangeOf(DimA)
        val r1 = doIt(m1)
        assert(r1.value == 2)
      }

      it("should accept a matrix") {
        val m2 = Tensor(
          (1, 2, 3),
          (4, 5, 6)
        )
        val r2 = doIt(m2)
        assert(r2.value == Seq(3, 6))
      }
    }

    describe("1-arg apply, when used on a batched one-dim tensor with extra appended dim") {
      def doIt[B <: Shape, L1 <: Dim, S <: Shape, T <: DType](t: Tensor[S, T, CPU.type])(using b:Batched1[B, L1, S]): Tensor[B :* Dim.One, T, CPU.type] = {
        import b.given

        val appended = t.unsqueezeAfterEnd
        val r = appended(dim[L1] % Index.Last) // This removes the L1 dimension, leaving the unsqueezed Dim.One.

        val rType: Tensor[B :* Dim.One, T, CPU.type] = r

        // Tests for all DimOperator variants accepting batches:
        // FIXME assert these results for a matrix.
        appended.sumBy(dim[Dim.One])
        appended.maxBy(dim[Dim.One])
        t.cat(rType)(dim[L1])

        r
      }

      it("should accept a vector") {
        case object DimA extends Dim.Static[3L]
        val m1 = Tensor.arangeOf(DimA)
        val r1 = doIt(m1)
        assert(r1.value == Seq(2))
      }
    }

    describe("1-arg apply, when used on a batched two-dim tensor") {
      def doIt[B <: Shape, L1 <: Dim, L2 <: Dim, S <: Shape, T <: DType](t: Tensor[S, T, CPU.type])(using Batched[B, (L1, L2), S]): Tensor[B :* L2, T, CPU.type] = {
        val r = t(dim[L1] % Index.Last)
        val rType: Tensor[B :* L2, T, CPU.type] = r
        r
      }

      it("should accept a matrix") {
        val m2 = Tensor(
          (1, 2, 3),
          (4, 5, 6)
        )
        val r2 = doIt(m2)
        assert(r2.value == Seq(4, 5, 6))
      }

      it("should accept a 3D tensor") {
        val m3 = Tensor(
          (
            (1, 2, 3),
            (4, 5, 6),
          ),(
            (7, 8, 9),
            (10, 11, 12),
          )
        )
        val r3 = doIt(m3)
        assert(r3.value == Seq(
          Seq(4, 5, 6),
          Seq(10, 11, 12)
        ))
      }
    }

    describe("2-arg apply, when used on a batched two-dim tensor") {
      def doIt[B <: Shape, L1 <: Dim, L2 <: Dim, S <: Shape, T <: DType](t: Tensor[S, T, CPU.type])(using Batched[B, (L1, L2), S]): Tensor[B, T, CPU.type] = {
        val r = t(
          dim[L1] % Index.Last,
          dim[L2] % Index.First
        )
        val rType: Tensor[B, T, CPU.type] = r
        r
      }

      it("should accept a matrix") {
        val m2 = Tensor(
          (1, 2, 3),
          (4, 5, 6)
        )
        val r2 = doIt(m2)
        assert(r2.value == 4)
      }

      it("should accept a 3D tensor") {
        val m3 = Tensor(
          (
            (1, 2, 3),
            (4, 5, 6),
          ),(
            (7, 8, 9),
            (10, 11, 12),
          )
        )
        val r3 = doIt(m3)
        assert(r3.value == Seq(4, 10))
      }
    }


    describe("cat") {
      it("can concatenate two equal matrices along the second dimension") {
        val m = Tensor.zeros(2L, 3L)
        val r = m.cat(m)((Select.Idx(1)))
        val rType: Tensor[(Static[2L], Dim.Dynamic), Float32, CPU.type] = r
        assert(r.size == Seq(2L, 6L))
      }

      it("can concatenate two unequal matrices along the first dimension") {
        val m1 = Tensor.zeros(1L, 3L)
        val m2 = Tensor.zeros(2L, 3L)
        val r = m1.cat(m2)((Select.Idx(0)))
        val rType: Tensor[(Dim.Dynamic, Static[3L]), Float32, CPU.type] = r
        assert(r.size == Seq(3L, 3L))
      }

      it("can concatenate two unequal matrices along the first dimension using ++") {
        val m1 = Tensor.zeros(1L, 3L)
        val m2 = Tensor.zeros(2L, 3L)
        val r = m1 ++ m2
        val rType: Tensor[(Dim.Dynamic, Static[3L]), Float32, CPU.type] = r
        assert(r.size == Seq(3L, 3L))
      }

      it("can concatenate two unequal matrices along the second dimension") {
        val m1 = Tensor.zeros(2L, 3L)
        val m2 = Tensor.zeros(2L, 1L)
        val r = m1.cat(m2)((Select.Idx(1)))
        val rType: Tensor[(Static[2L], Dim.Dynamic), Float32, CPU.type] = r
        assert(r.size == Seq(2L, 4L))
      }

      it("can concatenate into an existing dynamic output dimension") {
        class Row(size:Long) extends Dim.Dynamic(size)
        class Column(size:Long) extends Dim.Dynamic(size)
        val m = Tensor.zeros(2L, 3L).shaped[(Row, Column)]
        val r = m.cat(m)((Select.Idx(1)))
        val rType: Tensor[(Row, Column), Float32, CPU.type] = r
      }
    }

    describe("equal") {
      it("are two tensors with same type and contents") {
        val a = Tensor((1, 2, 3))
        val b = Tensor((1, 2, 3))
        assert(a.equal(b))
      }

      it("are not two tensors with same type and different contents") {
        val a = Tensor((1, 2, 3))
        val b = Tensor((1, 2, 4))
        assert(!a.equal(b))
      }

      it("are not two tensors with same type and different size") {
        val a = Tensor((1, 2, 3))
        val b = Tensor((1, 2))
        assert(!a.equal(b))
      }

      it("are not two tensors with same type and contents") {
        val a = Tensor((1, 2, 3))
        val b = Tensor((1.0, 2.0, 3.0))
        //assert(a.equal(b)) this won't even compile.
      }
    }

    describe("equals") {
      it("should equal tensors") {
        val a = Tensor((1, 2, 3))
        val b = Tensor((1, 2, 3))
        assert(a.equals(b))
      }

      it("should not equal tensors on different devices") {
        val a = Tensor((1, 2, 3)).to(Device.CUDA)
        val b = Tensor((1, 2, 3))
        assert(!a.equals(b))
      }

      it("should not equal tensors on different dtypes") {
        val a = Tensor((1, 2, 3)).to(float32)
        val b = Tensor((1, 2, 3)).to(int32)
        assert(!a.equals(b))
      }
    }

    describe("equals") {
      it("are two tensors with same type and contents") {
        val a = Tensor((1, 2, 3))
        val b = Tensor((1, 2, 3))
        assert(a == b)
      }

      it("are not two tensors with same type and different contents") {
        val a = Tensor((1, 2, 3))
        val b = Tensor((1, 2, 4))
        assert(a != b)
      }

      it("are not two tensors with same type and different size") {
        val a = Tensor((1, 2, 3))
        val b = Tensor((1, 2))
        assert(a != b)
      }

      it("are not two tensors with different type and contents") {
        val a = Tensor((1, 2, 3))
        val b = Tensor((1.0, 2.0, 3.0))
        assert(!a.equals(b))
        assert(a != b)
      }
    }

    describe("flatten") {
      it("can flatten a 1D tensor") {
        val t = Tensor((1, 2, 3))
        val r = t.flatten
        val rType: Tensor[Tuple1[Static[3L]], Int32, CPU.type] = r
        assert(r.size == Seq(3L))
        assert(r.value.toSeq == Seq(1, 2, 3))
      }

      it("can flatten a matrix") {
        val t = Tensor((
          ((1.0, 2.0, 3.0)),
          ((4.0, 5.0, 6.0))
        ))
        val tType: Tensor[(Static[2L], Static[3L]), Float64, CPU.type] = t
        val r = t.flatten
        val rType: Tensor[Tuple1[Static[2L] * Static[3L]], Float64, CPU.type] = r
      }
    }

    describe("maskedFill") {
      it("can fill elements of a float vector") {
        val t = Tensor((1.0, 2.0, 3.0))
        t.maskedFill(Tensor((false, true, false)), 4.0)
        assert(t.value.toSeq == Seq(1.0, 4.0, 3.0))
      }

      it("can't fill elements of a float vector with a batch") {
        val t = Tensor((1.0, 2.0, 3.0))
        val m = Tensor((
          ((false, true, false)),
          ((true, false, true))
        ))
        // t.maskedFill_(m, 4.0) This won't compile, so that's good.
      }

      it("can fill elements of a batch with a vector") {
        val t = Tensor((
          ((1.0, 2.0, 3.0)),
          ((4.0, 5.0, 6.0))
        ))
        val m = Tensor((false, true, false))
        t.maskedFill(m, 0.0)
        assert(t.value.toSeq == Seq(
          Seq(1.0, 0, 3.0),
          Seq(4.0, 0, 6.0)
        ))
      }

    }

    describe("maskedFilled") {
      it("can fill elements of a float vector") {
        val t = Tensor((1.0, 2.0, 3.0))
        val res = t.maskedFilled(Tensor((false, true, false)), 4.0)
        assert(res.value.toSeq == Seq(1.0, 4.0, 3.0))
      }

      it("can fill elements of a float vector with a batch") {
        val t = Tensor((1.0, 2.0, 3.0))
        val m = Tensor((
          ((false, true, false)),
          ((true, false, true))
        ))
        val r = t.maskedFilled(m, 0.0)
        assert(r.value.toSeq == Seq(
          Seq(1.0, 0.0, 3.0),
          Seq(0.0, 2.0, 0.0)
        ))
      }

      it("can fill elements of batch with a vector") {
        val t = Tensor((
          ((1.0, 2.0, 3.0)),
          ((4.0, 5.0, 6.0))
        ))
        val m = Tensor((false, true, false))
        val r = t.maskedFilled(m, 0.0)
        assert(r.value.toSeq == Seq(
          Seq(1.0, 0, 3.0),
          Seq(4.0, 0, 6.0)
        ))
      }
    }

    describe("matmul (`@`)") {
      case object DimA extends Dim.Static[2L]
      case object DimB extends Dim.Static[3L]
      case object DimC extends Dim.Static[4L]

      it("can multiply two vectors") {
        val a = Tensor.zeros(DimA)
        val b = Tensor.zeros(DimA)
        val r = a `@` b
        val rType: Tensor[Scalar, Float32, CPU.type] = r
        assert(r.size == Seq())
      }

      it("can multiply two matrices") {
        val a = Tensor.zeros(DimA, DimB)
        val b = Tensor.zeros(DimB, DimC)
        val r = a.matmul(b)
        val rType: Tensor[(DimA.type, DimC.type), Float32, CPU.type] = r
        assert(r.size == Seq(DimA.size, DimC.size))
      }

      it("can multiply vector with matrix") {
        val a = Tensor.zeros(DimA)
        val b = Tensor.zeros(DimA, DimB)
        val r = a.matmul(b)
        val rType: Tensor[Tuple1[DimB.type], Float32, CPU.type] = r
        assert(r.size == Seq(DimB.size))
      }

      it("can multiply matrix with vector") {
        val a = Tensor.zeros(DimA, DimB)
        val b = Tensor.zeros(DimB)
        val r = a.matmul(b)
        val rType: Tensor[Tuple1[DimA.type], Float32, CPU.type] = r
        assert(r.size == Seq(DimA.size))
      }

      it("can multiply two batches of matrices") {
        val a = Tensor.zeros(1L, DimA, DimB)
        val b = Tensor.zeros(1L, DimB, DimC)
        val r = a.matmul(b)
        val rType: Tensor[(Static[1L], DimA.type, DimC.type), Float32, CPU.type] = r
        assert(r.size == Seq(1L, DimA.size, DimC.size))
      }

      it("can broadcast uneqeual batches of matrices") {
        val a = Tensor.zeros(1L, DimA, DimB)
        val b = Tensor.zeros(2L, DimB, DimC)
        val r = a.matmul(b)
        val rType: Tensor[(Static[2L], DimA.type, DimC.type), Float32, CPU.type] = r
        assert(r.size == Seq(2L, DimA.size, DimC.size))
      }

      it("can broadcast different-dimensional batches of matrices") {
        val a = Tensor.zeros(2L, DimA, DimB)
        val b = Tensor.zeros((Static(1L), Static(2L), DimB, DimC))
        val r = a.matmul(b)
        val rType: Tensor[(Static[1L], Static[2L], DimA.type, DimC.type), Float32, CPU.type] = r
        assert(r.size == Seq(1L, 2L, DimA.size, DimC.size))
      }

      it("can multiply a matrix batch with a vector") {
        val a = Tensor.zeros((Static(1L), Static(4L), DimA, DimB))
        val b = Tensor.zeros(DimB)
        val r = a.matmul(b)
        val rType: Tensor[(Static[1L], Static[4L], DimA.type), Float32, CPU.type] = r
        assert(r.size == Seq(1L, 4L, DimA.size))
      }

      it("can multiply vector with matrix batch") {
        val a = Tensor.zeros(DimA)
        val b = Tensor.zeros(2L, DimA, DimB)
        val r = a.matmul(b)
        val rType: Tensor[(Static[2L], DimB.type), Float32, CPU.type] = r
        assert(r.size == Seq(2L, DimB.size))
      }

      it("can multiply batch with matrix") {
        val a = Tensor.zeros(2L, DimA, DimB)
        val b = Tensor.zeros(DimB, DimC)
        val r = a.matmul(b)
        val rType: Tensor[(Static[2L], DimA.type, DimC.type), Float32, CPU.type] = r
        assert(r.size == Seq(2L, DimA.size, DimC.size))
      }

      it("can multiply matrix with batch") {
        val a = Tensor.zeros(DimA, DimB)
        val b = Tensor.zeros(2L, DimB, DimC)
        val r = a.matmul(b)
        val rType: Tensor[(Static[2L], DimA.type, DimC.type), Float32, CPU.type] = r
        assert(r.size == Seq(2L, DimA.size, DimC.size))
      }
    }

    describe("max") {
      it("can find the maximum of a tensor") {
        val r = Tensor((1, 2, 3)).max
        val rType: Tensor[Scalar, Int32, CPU.type] = r
        assert(r.size == Seq.empty)
        assert(r.value == 3)
      }
    }

    describe("maxBy") {
      class Row(size: Long) extends Dim.Dynamic(size)
      class Column(size: Long) extends Dim.Dynamic(size)
      val t = Tensor((
        ((1.0, 5.0, 3.0)),
        ((4.0, 2.0, 6.0))
      )).shaped[(Row, Column)]

      it("can find the maximum for each row in a matrix") {
        val r = t.maxBy(dim[Row])
        val v = r.result
        val vType: Tensor[Column *: EmptyTuple.type, Float64, CPU.type] = v
        assert(v.size == Seq(3L))
        assert(v.value === Seq(4.0, 5.0, 6.0))

        val i = r.indices
        val iType: Tensor[Column *: EmptyTuple.type, Int64, CPU.type] = i
        assert(i.size == Seq(3L))
        assert(i.value === Seq(1, 0, 1))
        assert(i.dtype == int64)
      }

      it("can find the maximum for each column in a matrix") {
        val r = t.maxBy(dim[Column])
        val v = r.result
        val vType: Tensor[Row *: EmptyTuple.type, Float64, CPU.type] = v
        assert(v.size == Seq(2L))
        assert(v.value === Seq(5.0, 6.0))
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
        val resType: Tensor[DimB.type *: EmptyTuple, Float32, CPU.type] = res
        assert(res.size == Seq(3L))
        assert(res.value.toSeq == Seq(2.5, 0, 0))
      }

      it("can calculate mean of second dim") {
        var t = Tensor.zeros(DimA, DimB)
        t((0,0)) = 3.0
        t((1,0)) = 2.0
        val res = t.meanBy(DimB)
        val resType: Tensor[DimA.type *: EmptyTuple, Float32, CPU.type] = res
        assert(res.size == Seq(2L))
        assert(res.value.toSeq === Seq(1.0, 0.6666))
      }

      it("can calculate mean of selected dim and keep it") {
        var t = Tensor.zeros(DimA, DimB)
        t((0,0)) = 3.0
        t((1,0)) = 2.0
        val res = t.meanBy.keepDim(DimA)
        val resType: Tensor[(Dim.One, DimB.type), Float32, CPU.type] = res
        assert(res.size == Seq(1L, 3L))
        assert(res.value.toSeq == Seq(Seq(2.5, 0, 0)))
      }

      it("can calculate the full mean") {
        var t = Tensor.zeros(DimA, DimB)
        t((0,0)) = 3.0
        t((1,0)) = 2.0
        val res = t.mean
        val resType: Tensor[EmptyTuple, Float32, CPU.type] = res
        assert(res.value === 0.8333)
      }
    }

    describe("multinomial") {
      it("can pick values from a vector distribution") {
        val d = Tensor((0.1, 0.8, 0.1))
        val r = d.multinomial(Dim.Static(2L))
        val rType: Tensor[Tuple1[Static[2L]], Int64, CPU.type] = r
        assert(r.dtype == int64)
        assert(r.size == Seq(2L))
        assert(r.value === Seq(1, 2))
      }

      it("can pick values from a matrix distribution") {
        val d = Tensor((
          ((0.1, 0.8, 0.1)),
          ((0.1, 0.1, 0.8)),
        ))
        val r = d.multinomial(Dim.Static(2L))
        val rType: Tensor[(Static[2L], Static[2L]), Int64, CPU.type] = r
        assert(r.dtype == int64)
        assert(r.size == Seq(2L, 2L))
        assert(r.value === Seq(Seq(1, 2), Seq(2, 1)))
      }

      it("can pick a single value from a vector distribution") {
        val d = Tensor((0.1, 0.8, 0.1))
        val r = d.multinomial
        val rType: Tensor[EmptyTuple.type, Int64, CPU.type] = r
        assert(r.dtype == int64)
        assert(r.size.isEmpty)
        assert(r.value === 1)
      }

      it("can pick a single value from a matrix distribution") {
        val d = Tensor((
          ((0.1, 0.8, 0.1)),
          ((0.1, 0.1, 0.8)),
        ))
        val r = d.multinomial
        val rType: Tensor[Tuple1[Static[2L]], Int64, CPU.type] = r
        assert(r.dtype == int64)
        assert(r.size == Seq(2L))
        assert(r.value === Seq(1, 2))
      }
    }

    describe("padTo") {
      it("can pad a vector to a higher length") {
        case object DimA extends Dim.Static[4L]
        val r = Tensor((1, 2)).padTo(DimA, 9)
        val rType: Tensor[DimA.type *: EmptyTuple, Int32, CPU.type] = r
        assert(r.size == Seq(4L))
        assert(r.value.toSeq == Seq(1, 2, 9, 9))
      }

      it("can pad a vector to the same length") {
        case object DimA extends Dim.Static[2L]
        val r = Tensor((1, 2)).padTo(DimA, 9)
        val rType: Tensor[DimA.type *: EmptyTuple, Int32, CPU.type] = r
        assert(r.size == Seq(2L))
        assert(r.value.toSeq == Seq(1, 2))
      }
    }

    describe("plus") {
      it("can add a primitive") {
        val t = Tensor((1, 2, 3))
        val r = t + 1
        val rType: Tensor[Tuple1[Static[3L]], Int32, CPU.type] = r
        assert(r.size == Seq(3L))
        assert(r.value.toSeq == Seq(2, 3, 4))
      }

      it("can add vector and scalar") {
        val a = Tensor((1, 2, 3))
        val b = Tensor(1)
        val r = a + b
        val rType: Tensor[Tuple1[Static[3L]], Int32, CPU.type] = r
        assert(r.size == Seq(3L))
        assert(r.value.toSeq == Seq(2, 3, 4))
      }

      it("can add vectors of different lengths") {
        val a = Tensor((1, 2, 3))
        val b = Tensor(Tuple1(1))
        val r = a + b
        val rType: Tensor[Tuple1[Static[3L]], Int32, CPU.type] = r
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
        val rType: Tensor[(Static[4L], Static[4L]), Int32, CPU.type] = r
        assert(r.size == Seq(4L, 4L))
        assert(r.value.toSeq == Seq(
          Seq(6, 7, 8, 9),
          Seq(7, 8, 9, 10),
          Seq(8, 9, 10, 11),
          Seq(9, 10, 11, 12))
        )
      }

      it("can add a vector to a matrix with unknown dimensions") {
        val DimA = Dim.Dynamic(4)
        val DimB1 = Dim.Dynamic(4)
        val DimB2 = Dim.Dynamic(1) // 2 will throw a runtime exception in pytorch here.
        val a = Tensor.zeros(DimA)
        val b = Tensor.zeros(DimB1, DimB2)
        val r = a + b
        assert(r.size == Seq(4L, 4L))
      }

      it("can add a known-dim tensor to a valid batch") {
        import Tuple.++
        def doIt[B <: Shape, L1 <: Dim, S <: Shape, T <: DType](t: Tensor[S, T, CPU.type])(using b:Batched1[B, L1, S]): Tensor[B ++ (L1, Static[1L]), Promoted[T, Float32], CPU.type] = {
          val appended = t.unsqueezeAfterEnd
          val toAdd = Tensor.ones(1L, 1L)

          appended + toAdd
        }

        val r1 = doIt(Tensor.ones(3L))
        val r1Type: Tensor[(Static[3L], Static[1L]), Float32, CPU.type] = r1
        assert(r1.size == Seq(3L, 1L))
        assert(r1.value.toSeq == Seq(
          Seq(2.0),
          Seq(2.0),
          Seq(2.0)
        ))

        val r2 = doIt(Tensor.ones(3L, 2L))
        val r2Type: Tensor[(Static[3L], Static[2L], Static[1L]), Float32, CPU.type] = r2
        assert(r2.size == Seq(3L, 2L, 1L))
        assert(r2.value.toSeq == Seq(
          Seq(Seq(2.0), Seq(2.0)),
          Seq(Seq(2.0), Seq(2.0)),
          Seq(Seq(2.0), Seq(2.0))
        ))
      }

      it("can add a scalar to a valid batch") {
        def doIt[B <: Shape, L1 <: Dim, S <: Shape](t: Tensor[S, Float32, CPU.type])(using b:Batched1[B, L1, S]) = {
          val appended = t.unsqueezeAfterEnd

          appended + 5.0
        }

        val r1 = doIt(Tensor.ones(3L))
        val r1Type: Tensor[(Static[3L], One), Float32, CPU.type] = r1
        assert(r1.size == Seq(3L, 1L))
        assert(r1.value.toSeq == Seq(
          Seq(6.0),
          Seq(6.0),
          Seq(6.0)
        ))
      }


    }

    describe("sizeOf") {
      class DimA(size: Long) extends Dim.Dynamic(size)
      class DimB(size: Long) extends Dim.Dynamic(size)
      val t = Tensor.zeros(DimA(2), DimB(3))

      it("can return the size of a selected dimension") {
        assert(t.sizeOf(First).size == 2L)
        assert(t.sizeOf(dim[DimA]).size == 2L)
        assert(t.sizeOf(dim[DimB]).size == 3L)
      }

      it("can reify a dimension") {
        val size = t.sizeOf(DimA(_))
        val sizeT: DimA = size
        assert(size.size == 2)
      }
    }

    describe("stackMap") {
      it("can turn a vector into a matrix through a lambda") {
        case object DimA extends Dim.Static[3L]
        case object DimB extends Dim.Static[2L]
        val m = Tensor.zeros(DimA).stackMap(v => Tensor.zeros(DimB))
        val mType: Tensor[(DimA.type, DimB.type), Float32, CPU.type] = m
        assert(m.size == Seq(DimA.size, DimB.size))
      }
    }

    describe("std") {
      it("can calculate standard deviation") {
        var t = Tensor((1.0, 2.0, 3.0))
        val res = t.stdBy(First)
        val resType: Tensor[EmptyTuple, Float64, CPU.type] = res
        assert(res.size == Seq())
        assert(res.value == 1.0)
      }
    }

    describe("squeeze") {
      it("can remove a dimension of one") {
        val t = Tensor((
          Tuple1(1),
          Tuple1(2),
          Tuple1(3)
        ))
        val r = t.squeeze(Last) // Select.First gives a nice compile error.
        val rType: Tensor[Tuple1[Static[3L]], Int32, CPU.type] = r
        assert(r.size == Seq(3L))
      }
    }

    describe("sum") {
      val t = Tensor((
        ((1.0, 2.0, 3.0)),
        ((4.0, 5.0, 6.0))
      ))

      it("can sum all dimensions of a matrix") {
        val s = t.sum.value
        assert(s === 21.0)
      }

      it("can sum across a single dimension") {
        val s = t.sumBy(Select.Idx(0))
        val sType: Tensor[Static[3L] *: EmptyTuple.type, Float64, CPU.type] = s
        assert(s.size == Seq(3L))
        assert(s.value.toSeq === Seq(5.0, 7.0, 9.0))
      }
    }

    describe("summary") {
      it("can summarize all types") {
        assert(Tensor(5).to(int8).summary == "5")
        assert(Tensor(5, 1).to(uint8).summary == "(5, 1)")
        assert(Tensor(5, 1).to(int16).summary == "(5, 1)")
        assert(Tensor(5, 1).to(int32).summary == "(5, 1)")
        assert(Tensor(5, 1).to(int64).summary == "(5, 1)")
        assert(Tensor(5.0, 1.0).to(float16).summary == "(5.0000, 1.0000)")
        assert(Tensor(5.0, 1.0).to(float32).summary == "(5.0000, 1.0000)")
        assert(Tensor(5.0, 1.0).to(float64).summary == "(5.0000, 1.0000)")
      }
    }

    describe("to") {
      it("can copy a tensor to the GPU and back") {
        val c = Tensor((1, 2, 3))
        val g = c.to(CUDA)
        // val r = g + Tensor((4, 5, 6)) // won't compile, since tensors aren't on the same device.
        val r = g + Tensor((4, 5, 6)).to(CUDA)
        val v = r.to(CPU).value // r.value won't compile, since non-CPU tensors can't be read directly.
        assert(v.toSeq == Seq(5, 7, 9))
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
        val aType: Tensor[(Static[2L], Static[2L], Static[3L]), Int32, CPU.type] = a
        val b = a.transpose(Select.Idx(0), Select.Idx(2))
        val bType: Tensor[(Static[3L], Static[2L], Static[2L]), Int32, CPU.type] = b
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
    }

    describe("t") {
      it("can transpose a matrix") {
        val m = Tensor(
          Tuple1(
            ((1, 2))
          )
        )
        val r = m.t
        val rType: Tensor[(Static[2L], Static[1L]), Int32, CPU.type] = r
        assert(r.size == Seq(2L, 1L))
        assert(r.value === Seq(
          Seq(1),
          Seq(2)
        ))
      }

      it("can transpose a batched matrix") {
        val m = Tensor.zeros(1L, 2L, 3L)
        val r = m.t
        val rType: Tensor[(Static[1L], Static[3L], Static[2L]), Float32, CPU.type] = r
        assert(r.size == Seq(1L, 3L, 2L))
      }
    }

    describe("shaped") {
      class MyDim(size: Long) extends Dim.Dynamic(size)
      it("can put a dimension on a vector without using a tuple") {
        val t = Tensor((1, 2, 3)).shaped[MyDim]
        val tType = t
      }

      it("can put dimensions on a 3d tensor") {
        val t = Tensor(Tuple1(Tuple1(Tuple1(42)))).shaped[(MyDim, MyDim, MyDim)]
        val tType: Tensor[(MyDim, MyDim, MyDim), Int32, CPU.type] = t
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
        a.value = 4
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
        val r = vector.unsqueezeAfter(Last)
        val rType: Tensor[(DimA.type, Static[1L]), Float32, CPU.type] = r
        assert(r.size == Seq(2L, 1L))
        assert(r.value.toSeq == Seq(Seq(0), Seq(0)))
      }

      it("can unsqueeze after the last dim of a matrix") {
        // Verify that we can unsqueeze by type as well
        val r2 = matrix.unsqueezeAfter(dim[DimB.type])
        val r2Type: Tensor[(DimA.type, DimB.type, Static[1L]), Float32, CPU.type] = r2

        val r = matrix.unsqueezeAfter(DimB)
        val rType: Tensor[(DimA.type, DimB.type, Static[1L]), Float32, CPU.type] = r
        assert(r.size == Seq(2L, 3L, 1L))
        assert(r.value.toSeq == Seq(
          Seq(Seq(0),Seq(0),Seq(0)),
          Seq(Seq(0),Seq(0),Seq(0))
        ))
      }

      it("can unsqueeze after the first dim of a matrix") {
        val r = matrix.unsqueezeAfter(DimA)
        val rType: Tensor[(DimA.type, Static[1L], DimB.type), Float32, CPU.type] = r
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
        val r = vector.unsqueezeBefore(First)
        val rType: Tensor[(Static[1L], DimA.type), Float32, CPU.type] = r
        assert(r.size == Seq(1L, 2L))
        assert(r.value.toSeq == Seq(Seq(0, 0)))
      }

      it("can unsqueeze before first (by index)") {
        val r = vector.unsqueezeBefore(Select.Idx(0))
        val rType: Tensor[(Static[1L], DimA.type), Float32, CPU.type] = r
        assert(r.size == Seq(1L, 2L))
        assert(r.value.toSeq == Seq(Seq(0, 0)))
      }

      it("can unsequeeze before first dim of a matrix") {
        val r = matrix.unsqueezeBefore(DimA)
        val rType: Tensor[(Static[1L], DimA.type, DimB.type), Float32, CPU.type] = r
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
        val rType: Tensor[(DimA.type, Static[1L], DimB.type), Float32, CPU.type] = r
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

      it("can unsqueeze a scalar into a vector") {
        val t = Tensor(1.0)
        val r = t.unsqueeze
        val rType: Tensor[Tuple1[One], Float64, CPU.type] = r
        assert(r.size == Seq(1L))
        assert(r.value.toSeq === Seq(1.0))
      }
    }

    describe("split and unsplit") {
      case object DimA extends Static[6L]
      case object DimB extends Static[3L]

      it("can split on specific dim and then unsplit") {
        val matrix = Tensor.zeros(DimA, DimB)
        matrix((1, 1)) = 1.0
        val res = matrix.view.split(DimA).into(Dim.Static(2L))
        val resType: Tensor[(Static[2L], DimA.type / Dim.Static[2L], DimB.type), Float32, CPU.type] = res
        assert(res.size == Seq(2L, 3L, 3L))
        assert(res.value(0)(1)(1) == 1.0)

        val un = res.view.merge[DimA.type / Dim.Static[2L]]
        assert(un.size == Seq(6L, 3L))
        assert(un.value(1)(1) == 1.0)
      }

      it("can unsplit on a specific dim and then split") {
        val matrix = Tensor.zeros(DimA, DimB)
        matrix((1, 1)) = 1.0
        val res = matrix.view.merge(DimB)
        val resType: Tensor[Tuple1[DimA.type * DimB.type], Float32, CPU.type] = res
        assert(res.size == Seq(DimA.size * DimB.size))
        assert(res.value.toSeq == Seq(
          0,0,0,
          0,1,0,
          0,0,0,
          0,0,0,
          0,0,0,
          0,0,0
        ).map(_.toFloat))

        val spl = res.view.split(dim[DimA.type * DimB.type]).into(DimA)
        val splType: Tensor[(DimA.type, DimB.type), Float32, CPU.type] = spl
        assert(spl.size == Seq(DimA.size, DimB.size))

        // Test that we can swap the dimensions by splitting into the other dimension
        val spl2 = res.view.split(dim[DimA.type * DimB.type]).into(DimB)
        val spl2Type: Tensor[(DimB.type, DimA.type), Float32, CPU.type] = spl2
        assert(spl2.size == Seq(DimB.size, DimA.size))
      }

      it("can split on last") {
        case object DimC extends Static[4L]
        val t = Tensor.zeros(DimA, DimB, DimC)
        val res = t.view.split(DimC).into(Dim.Static(4L))
        val resType: Tensor[(DimA.type, DimB.type, Static[4L], DimC.type / Dim.Static[4L]), Float32, CPU.type] = res
        assert(res.size == Seq(6L, 3L, 4L, 1L))

        val un = res.view.merge[DimC.type / Dim.Static[4L]]
        assert(un.size == Seq(6L, 3L, 4L))
      }
    }
  }
}
