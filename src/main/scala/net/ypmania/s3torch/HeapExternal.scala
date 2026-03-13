package net.ypmania.s3torch

import org.bytedeco.javacpp.Pointer
import scala.util.Using
import org.bytedeco.javacpp.PointerScope

trait HeapExternal[T] {
  def toPointer(t: T): Pointer
}

object HeapExternal {
  given HeapExternal[Unit] with { def toPointer(u: Unit) = null }
  given HeapExternal[Pointer] with { def toPointer (p: Pointer) = p }
  given tensor[S <: Tuple, T <: DType, D <: Device]: HeapExternal[Tensor[S, T, D]] with { def toPointer (t: Tensor[S, T, D]) = t.native }

  /** Runs the given block, removing all allocated pointers afterwards. T itself however is preserved, and put into any
    * enclosing scope (if one is open).
    */
  def scoped[T: HeapExternal](block: =>T): T = {
    val r = Using.resource(new PointerScope()) { scope =>
      val result = block
      val ptr = summon[HeapExternal[T]].toPointer(result)
      if (ptr != null) {
        ptr.retainReference()
      }
      result
    }
    val ptr = summon[HeapExternal[T]].toPointer(r)
    val scope = PointerScope.getInnerScope()
    if (scope != null && ptr != null) {
      scope.attach(ptr)
    }
    if (ptr != null) {
      ptr.releaseReference()
    }
    r
  }
}
