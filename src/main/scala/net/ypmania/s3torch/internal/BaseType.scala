package net.ypmania.s3torch.internal

trait BaseType[V] {
  type Out
}

trait BaseTypePrio0 {
  given [V]: BaseType[V] with { type Out = V }
}

object BaseType extends BaseTypePrio0 {
  given [V, S <: Seq[V]](using base: BaseType[V]): BaseType[S] with { type Out = base.Out }
  given [T <: Tuple](using base: BaseType[Tuple.Union[T]]): BaseType[T] with { type Out = base.Out }
}
