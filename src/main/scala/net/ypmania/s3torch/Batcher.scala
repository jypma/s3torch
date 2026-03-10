package net.ypmania.s3torch

/** Creates a "batcher" utility object, that can create ad-hoc batches by invoking ".stack" on transformed values of [T], e.g.
  * case class BatchSize(size: Long) extends Dim
  * val batcher = Tensor.batcher(BatchSize(_), myCollection)
  * val inputs = batcher(_.input) // Combines the "input" tensors into a batch of the size of "myCollection", typed BatchSize
  * val outputs = batcher(_.output)
  */
case class Batcher[A, B <: Dim](private val mkDim: Long => B, private val collection: Iterable[A]) {
  /** Maps the next batch's [A] in the collection to the given function, and returns the results as a batched Tensor. */
  def apply[S <: Tuple, T <: DType, D <: Device](getTensor: A => Tensor[S, T, D]): Tensor[B *: S, T, D] = {
    Tensor.stack[B](collection.map(getTensor))
  }

  /** Returns the size of the next batch (as a typed dimension) */
  def size: B = mkDim(collection.size)
}
