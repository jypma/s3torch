package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Dim
import  scala.compiletime.ops.long.*

type AddDim[D1, D2] = (D1, D2) match {
  case (Dim.Static[s1], Dim.Static[s2]) => Dim.Static[s1 + s2]
  case _ => Dim.Dynamic
}
