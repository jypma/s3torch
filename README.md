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
- It calculates the resulting shape of [matrix multiplication](src/test/scala/net/ypmania/s3torch/TensorSpec.scala#L467), again including [broadcasting rules](src/test/scala/net/ypmania/s3torch/TensorSpec.scala#L483).
- You can refer to dimensions by a logical name (type), instead of just index, e.g. when calculating [`meanBy`](src/test/scala/net/ypmania/s3torch/TensorSpec.scala#L362).
- Transposing dimensions is visible [in the return type](src/test/scala/net/ypmania/s3torch/TensorSpec.scala#L649)


