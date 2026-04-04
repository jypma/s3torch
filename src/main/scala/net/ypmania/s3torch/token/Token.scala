package net.ypmania.s3torch.token

/** Type class for token-like integers */
trait Token[T] {
  /** The token that represents an unknown input or output */
  def unknown: T
  /** The token that ordinally follows the given one */
  def next(t: T): T
  /** The maximum from a collection of tokens. */
  def max(ts: Iterable[T]): T
}
