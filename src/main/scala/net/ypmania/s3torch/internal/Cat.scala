package net.ypmania.s3torch.internal

import net.ypmania.s3torch.Dim

import scala.compiletime.ops.int._

trait Cat[S <: Tuple, U <: Tuple, Idx <: Int] {}

object Cat {
  given uneq0[D1, D2, T1 <: Tuple, T2 <: Tuple]: Cat[D1 *: T1, D2 *: T2, 0] with {}
  given uneq[D1, D2, T1 <: Tuple, T2 <: Tuple, Idx <: Int](using Cat[T1, T2, Idx - 1]): Cat[D1 *: T1, D2 *: T2, Idx] with {}

  trait PickDynamic[D1, D2] {
    type Out
  }
  trait PickDynamicPrio0 {
    given fallback[D1, D2]: PickDynamic[D1, D2] with {
      type Out = Dim.Dynamic
    }
  }
  trait PickDynamicPrio1 extends PickDynamicPrio0 {
    given right[D1, D2](using D2 <:< Dim.Dynamic): PickDynamic[D1, D2] with {
      type Out = D2
    }
  }
  object PickDynamic extends PickDynamicPrio1 {
    given left[D1, D2](using D1 <:< Dim.Dynamic): PickDynamic[D1, D2] with {
      type Out = D1
    }
  }
}
