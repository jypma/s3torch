package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Tensor.*
import net.ypmania.s3torch.*

trait StaticShape[S] {
  type OutputShape <: Tuple
  def size: Seq[Long]
}

object StaticShape {
  import Dim.*

  type StaticDim[D] = D match {
    case Singleton & Long => Static[D]
    case Static[size] => D
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

  given [D](using v:ValueOf[StaticSize[D]]): StaticShape[D] with {
    type OutputShape = Tuple1[StaticDim[D]]
    def size = Seq(v.value)
  }

  given [D1, D2](using v1:ValueOf[StaticSize[D1]], v2:ValueOf[StaticSize[D2]]): StaticShape[(D1, D2)] with {
    type OutputShape = (StaticDim[D1], StaticDim[D2])
    def size = Seq(v1.value, v2.value)
  }
}
