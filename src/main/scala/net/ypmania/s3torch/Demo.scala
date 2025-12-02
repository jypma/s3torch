package net.ypmania.s3torch

object Demo {
  /** One dimension of a tensor, which might be statically known at compile time. */
  trait Dim
  trait Static[S <: Singleton & Long] extends Dim
  case class Tensor[S <: Tuple](size: Seq[Long])

  /** Type class that proves that a certain type is a valid representation of a statically-known dimension. */
  trait StaticDim[D] {
    type OutputDim <: Dim
    def size: Long
  }

  object StaticDim {
    given [S <: Singleton & Long](using ValueOf[S]): StaticDim[S] with {
      type OutputDim = Static[S]
      def size = valueOf[S]
    }
  }

  // Getting the given directly does work:
  trait StaticShapeV2[S] {
    type OutputShape <: Tuple
    def size: Seq[Long]
  }

  object StaticShapeV2 {
    given sizeOf1Dim[S <: Singleton & Long](using ValueOf[S]): StaticShapeV2[S] with {
      type OutputShape = Tuple1[S]
      def size = Seq(valueOf[S])
    }
  }

  def createv2[T](using s: StaticShapeV2[T]): Tensor[s.OutputShape] = Tensor(s.size)

  val v2 = createv2[10L] // Tensor[Tuple1[10L]]

  // However, getting the Dim through a second-level given doesn't work:
  trait StaticShape[S] {
    type OutputShape <: Tuple
    def size: Seq[Long]
  }

  object StaticShape {
    given sizeOf1Dim[D](using dim: StaticDim[D]): StaticShape[D] with {
      type OutputShape = Tuple1[dim.OutputDim]
      def size = Seq(dim.size)
    }
  }

  def create[T](using s: StaticShape[T]): Tensor[s.OutputShape] = Tensor(s.size)

  val v1 = create[10L] // Tensor[? >: Tuple1[Nothing] <: Tuple1[Dim]]
}
