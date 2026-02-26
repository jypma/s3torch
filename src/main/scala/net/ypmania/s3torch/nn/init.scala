package net.ypmania.s3torch.nn

import org.bytedeco.pytorch.global.torch
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.RandomSource

object init {
  def xavier_normal(t: Tensor[?, ?, ?])(using rnd: RandomSource): Unit = rnd {
    torch.xavier_normal_(t.native)
  }

  def xavier_normal(t: Tensor[?, ?, ?], gain: Double = 1.0)(using rnd: RandomSource): Unit = rnd {
    torch.xavier_normal_(t.native, gain)
  }

  def xavier_uniform(t: Tensor[?, ?, ?])(using rnd: RandomSource): Unit = rnd {
    torch.xavier_uniform_(t.native)
  }

  def xavier_uniform(t: Tensor[?, ?, ?], gain: Double)(using rnd: RandomSource): Unit = rnd {
    torch.xavier_uniform_(t.native, gain)
  }
}
