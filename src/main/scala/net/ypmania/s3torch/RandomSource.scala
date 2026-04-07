package net.ypmania.s3torch

import org.bytedeco.pytorch

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
  given live: RandomSource = new  RandomSource {
    def apply[T](fn: => T) = fn
  }

  /** A random source that runs each block with the same fixed seed. */
  def fixedSeed(seed: Long): RandomSource = new RandomSource {
    override def apply[T](fn: => T) = withSeed(seed) { fn }

    override def fork: RandomSource = new RandomSource {
      var seed = 0L
      def apply[T](fn: => T) = withSeed(seed) {
        seed += 1
        fn
      }
    }

    private def withSeed[T](seed: Long)(fn: => T): T = {
      RandomSource.synchronized {
        pytorch.global.torch.manual_seed(seed)
        fn
      }
    }
  }
}
