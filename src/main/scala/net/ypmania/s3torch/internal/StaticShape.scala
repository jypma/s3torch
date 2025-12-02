package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Tensor.*
import net.ypmania.s3torch.*

/*
trait StaticDim[D] {
  type OutputDim <: Dim
  def size: Long
}

object StaticDim {

  given [S <: Singleton & Long](using ValueOf[S]): StaticDim[S] with {
    type OutputDim = Static[S]
    def size = valueOf[S]
   }


  /*
  given [S <: Singleton & Long, T <: Static[S]](using ValueOf[S]): StaticDim[T] with {
    type OutputDim = T
    def size = valueOf[S]
   }
   */
  // TODO Find out why things break if we also allow Tuple1[Static[10L]] as a Static Dim
}
 */

trait StaticShape[S] {
  type OutputShape <: Tuple
  def size: Seq[Long]
}

object StaticShape {
  import Dim.*

  type StaticDim[D] = D match {
    case Singleton & Long => Static[D]
    case Static[size] => D
    case Tuple1[Static[size]] => Static[size] // FIXME remove or match the subtype somehow
  }

  type StaticSize[D] <: Long = D match {
    case Static[size] => size
    case Tuple1[Static[size]] => size
    case 1L => 1L
    case 10L => 10L
  }

  given StaticShape[Scalar] with {
    type OutputShape = Scalar
    def size = Seq.empty
  }

  /*
   // StaticDim
  given sizeOf1Dim[D](using dim: StaticDim[D]): StaticShape[D] with {
    type OutputShape = Tuple1[dim.OutputDim]
    def size = Seq(dim.size)
  }


  given sizeOf1Tup[S](using dim: StaticDim[S]): StaticShape[Tuple1[S]] with {
    type OutputShape = Tuple1[dim.OutputDim]
    def size = Seq(dim.size)
  }
   */

  /*
   // Direct
  given sizeOf1Lit[S <: Singleton & Long](using ValueOf[S]): StaticShape[S] with {
    type OutputShape = Tuple1[Static[S]]
    def size = Seq(valueOf[S])
  }

  given sizeOf1Dim[S <: Singleton & Long, T <: Static[S]](using ValueOf[S]): StaticShape[T] with {
    type OutputShape = Tuple1[T]
    def size = Seq(valueOf[S])
  }

  given sizeOf1Tup[S <: Singleton & Long, T <: Static[S]](using ValueOf[S]): StaticShape[Tuple1[T]] with {
    type OutputShape = Tuple1[T]
    def size = Seq(valueOf[S])
  }
   */

  // match type
  given sizeOf1Mat[D](using v:ValueOf[StaticSize[D]]): StaticShape[D] with {
    type OutputShape = Tuple1[StaticDim[D]]
    def size = Seq(v.value)
  }

  given sizeOf2Mat[
    D1,
    D2]
  (using v1:ValueOf[StaticSize[D1]], v2:ValueOf[StaticSize[D2]]): StaticShape[(D1, D2)] with {
    type OutputShape = (StaticDim[D1], StaticDim[D2])
    def size = Seq(v1.value, v2.value)
  }
  /*
  given sizeOf2[
    S1 <: Singleton & Long, T1 <: Static[S1],
    S2 <: Singleton & Long, T2 <: Static[S2]
  ](using ValueOf[S1], ValueOf[S2]): StaticShape[(T1, T2)] with {
    type OutputShape = (T1, T2)
    def size = Seq(valueOf[S1], valueOf[S2])
 }
   */
  /*
  given sizeOf2[
    D1,
    S2 <: Singleton & Long, T2 <: Static[S2]
  ](using d1: StaticDim[D1], v2: ValueOf[S2]): StaticShape[(D1, T2)] with {
    type OutputShape = (d1.OutputDim, T2)
    def size = Seq(d1.size, valueOf[S2])
 }
   */
}
