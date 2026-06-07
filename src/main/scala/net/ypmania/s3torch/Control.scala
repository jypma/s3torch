package net.ypmania.s3torch

object Control {
  /** Executes the predicate, and if it is defined, executed the loop on it. Repeats until the predicate returns false. */
  def whileDefined[I](pred: => Option[I])(loop: I => Unit): Unit = {
    var res = pred
    while (res.isDefined) {
      loop(res.get)
      res = pred
    }
  }
}
