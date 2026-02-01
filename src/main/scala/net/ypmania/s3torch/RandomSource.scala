package net.ypmania.s3torch

/** Wrapper trait that can wrap functions that use a source of
  * randomness. This allows tests to set a reproducable random
  * seed. We don't support libtorch's "Generator" concept, since
  * libtorch's built-in neural network modules (nn.*) always use the
  * global random generator anyway. */
trait RandomSource() {
  def apply[T](fn: => T): T

  /** Returns a new, potentially stateful RandomSource, that creates its own predictable sequence of numbers */
  def fork: RandomSource = this
}

object RandomSource {
  /** The default random source does not impose any explicit behavior. */
  val live = new  RandomSource {
    def apply[T](fn: => T) = fn
  }
}
