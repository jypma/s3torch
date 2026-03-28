# s3torch

This is a Scala library on top of `libtorch`. It provides full dimensional type safety for its `Tensor` type.

This started out as a branch of [storch](https://github.com/bytedeco/storch), but currently is its own thing, to allow more freedom in re-modeling the `Tensor` class without having to fix a lot of usages.

## Dimension

A dimension is represented by the type `Dim`. Anything that extends `Dim` can represent a dimension. Dimensions are usually referred to by their type. 
```scala
trait Dim {
  def size: Long
}
```

In addition, if a dimension is known at compile time, more precise tensor operations are available, that will track resulting dimensionalities through the code base. These compile-time known dimensions extend `Dim.Static`.
```scala
abstract class Static[S <: Long](using ValueOf[S]) extends Dim {
  // ...
}
```

So, for example, you could specify the dimensions for a matrix with an unknown number of rows, but known number of columns:

```scala
case class Rows(size: Long) extends Dim
case object Columns extends Static[10L]
```

## Tensor

A tensor has the following type signature:
```scala
class Tensor[S <: Tuple, T <: DType]
```

where
- `T` is the data type. `DType` is modeled, much like `storch`, as a simple enumeration-like sealed trait with entries like `Float32` or `Int8`.
- `S` is the "shape", or dimensions, of the tensor. This is a `Tuple`, where each element must be a subclass of `Dim`. 

## Examples of type safety

There are many ways in which having dimensions available helps development of code. The examples below link to working example code in `TensorSpec.scala`.

- Pytorch's [broadcasting rules](https://docs.pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics) are  automatically [applied and checked](src/test/scala/net/ypmania/s3torch/TensorSpec.scala#L559).
```scala
        val a = Tensor((1, 2, 3, 4)) // [4]
        val b = Tensor((             // [4, 1]
          Tuple1(5),
          Tuple1(6),
          Tuple1(7),
          Tuple1(8)
        ))
        val r = a + b                // [4, 4]
        val rType: Tensor[(Static[4L], Static[4L]), Int32, CPU.type] = r
```

- It calculates the resulting shape of [matrix multiplication](src/test/scala/net/ypmania/s3torch/TensorSpec.scala#L467), again including [broadcasting rules](src/test/scala/net/ypmania/s3torch/TensorSpec.scala#L483).
```scala
      it("can multiply matrix with vector") {
        val a = Tensor.zeros(DimA, DimB)
        val b = Tensor.zeros(DimB)
        val r = a `@` b
        val rType: Tensor[Tuple1[DimA.type], Float32, CPU.type] = r
        assert(r.size == Seq(DimA.size))
      }

      it("can multiply batch with matrix") {
        val a = Tensor.zeros(2L, DimA, DimB)
        val b = Tensor.zeros(DimB, DimC)
        val r = a `@` b
        val rType: Tensor[(Static[2L], DimA.type, DimC.type), Float32, CPU.type] = r
        assert(r.size == Seq(2L, DimA.size, DimC.size))
      }
```

- You can refer to dimensions by a logical name (type), instead of just index, e.g. when calculating [`meanBy`](src/test/scala/net/ypmania/s3torch/TensorSpec.scala#L362).
```scala
        case object DimA extends Dim.Static[2L]
        case object DimB extends Dim.Static[3L]
        var t = Tensor.zeros(DimA, DimB)
        t((0,0)) = 3.0
        t((1,0)) = 2.0
        val res = t.meanBy(DimA)
        val resType: Tensor[Tuple1[DimB.type], Float32, CPU.type] = res
```

- Transposing dimensions is visible [in the return type](src/test/scala/net/ypmania/s3torch/TensorSpec.scala#L649).
```scala
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
        val b = a.transpose(Shape.Select.Idx(0), Shape.Select.Idx(2))
        val bType: Tensor[(Static[3L], Static[2L], Static[2L]), Int32, CPU.type] = b
```

# Compiler bugs

```
java.lang.AssertionError: assertion failed: no owner from  <none>/ <none> 
```
The above assertion can happen if a `Tensor` with an inferred type is used in a subsequent calculation. Try to insert an explicit type for the tensor variable, or for any intermediate results. This has been hard to create minimal reproducer for, as it only occurs on a long chain of inferred/derived types, involving `given`.
