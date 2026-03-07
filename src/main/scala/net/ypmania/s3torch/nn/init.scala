package net.ypmania.s3torch.nn

import org.bytedeco.pytorch.global.torch
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.RandomSource
import net.ypmania.s3torch.Shape
import net.ypmania.s3torch.Shape.Size
import scala.compiletime.ops.int.>=

object init {
  def xavier_normal[S <: Shape](t: Tensor[S, ?, ?])(using rnd: RandomSource)(using Size[S] >= 2 =:= true): Unit = rnd {
    torch.xavier_normal_(t.native)
  }

  def xavier_normal[S <: Shape](t: Tensor[?, ?, ?], gain: Double = 1.0)(using rnd: RandomSource)(using Size[S] >= 2 =:= true): Unit = rnd {
    torch.xavier_normal_(t.native, gain)
  }

  def xavier_uniform[S <: Shape](t: Tensor[S, ?, ?])(using rnd: RandomSource)(using Size[S] >= 2 =:= true): Unit = rnd {
    torch.xavier_uniform_(t.native)
  }

  def xavier_uniform[S <: Shape](t: Tensor[S, ?, ?], gain: Double)(using rnd: RandomSource)(using Size[S] >= 2 =:= true): Unit = rnd {
    torch.xavier_uniform_(t.native, gain)
  }
}
