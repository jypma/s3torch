package net.ypmania.s3torch.transformer

import org.bytedeco.pytorch
import net.ypmania.s3torch
import net.ypmania.s3torch.Default
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.PaddingMode
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.internal.FromScala
import net.ypmania.s3torch.internal.FromScala.ToScalar

trait Token[T] {
  def unknown: T
  def next(t: T): T
  def max(ts: Iterable[T]): T
  def toInt(t: T): Int
}

object Token {
  given Token[Int] = new IntToken {}
}

trait IntToken extends Token[Int] {
  def unknown = 0
  def next(t: Int) = t + 1
  def max(ts: Iterable[Int]) = ts.max
  def toInt(t: Int) = t
}

trait LongToken extends Token[Long] {
  def unknown = 0
  def next(t: Long) = t + 1
  def max(ts: Iterable[Long]) = ts.max
  def toInt(t: Long) = t.toInt
}

trait IntTokenType {
  opaque type T = Int

  extension (t: T) {
    def toInt: Int = t
  }

  given Token[T] = new IntToken {}
  given ToScalar[T] = summon[ToScalar[Int]]

  abstract class DType extends s3torch.DType.Int32
  val dType = new DType {}

  /** Turns the given token into a scalar tensor */
  def toTensor[Dv <: Device](token: T)(using Default[Dv]): Tensor[EmptyTuple, DType, Dv] = {
    val int = token.asInstanceOf[Int] // Safe, because of opaque type
    Tensor(int, dType)
  }

  /** Turns the given tokens into a tensor */
  def toTensor[Dv <: Device](tokens: Seq[T])(using Default[Dv]): Tensor[Dim.Dynamic *: EmptyTuple, DType, Dv] = {
    val ints = tokens.asInstanceOf[Seq[Int]] // Safe, because of opaque type
    Tensor(ints, dType)
  }

  /** Turns the given tokens into a tensor, padded up to [D] with [pad], or None if the source is too long. */
  def toTensor[Dv <: Device, D <: Dim](tokens: Seq[T], dim: D, pad: T)(using Default[Dv]): Option[Tensor[D *: EmptyTuple, DType, Dv]] = {
    toTensor(tokens).padToOption(dim)(pad, PaddingMode.Append)
  }
}

trait LongTokenType {
  opaque type T = Long

  extension (t: T) {
    def toLong: Long = t
  }

  given Token[T] = new LongToken {}
  given ToScalar[T] = summon[ToScalar[Long]]

  abstract class DType extends s3torch.DType.Int64
  val dType = new DType {}

  /** Turns the given tokens into a tensor */
  def toTensor[Dv <: Device](tokens: Seq[T])(using Default[Dv]): Tensor[Dim.Dynamic *: EmptyTuple, DType, Dv] = {
    val ints = tokens.asInstanceOf[Seq[Long]] // Safe, because of opaque type
    Tensor(ints, dType)
  }

  /** Turns the given tokens into a tensor, padded up to [D] with [pad], or None if the source is too long.
    * This relies on pytorch's pad function, and hence may have errors for very large Long values.
    */
  def toTensor[Dv <: Device, D <: Dim](tokens: Seq[T], dim: D, pad: T)(using Default[Dv]): Option[Tensor[D *: EmptyTuple, DType, Dv]] = {
    toTensor(tokens).padToOption(dim)(pad.toDouble, PaddingMode.Append)
  }
}

abstract class Tokenizer[A: Token] {
  private val t = summon[Token[A]]

  def max: A
  def tokenize(in: String): Seq[A]

}

case class WordData[T](known: Map[String, T])

class WordTokenizer[A: Token](data: WordData[A]) extends Tokenizer[A] {
  private val t = summon[Token[A]]
  import WordTokenizer._
  private var nextReservedToken = t.max(data.known.values)

  protected def reservedToken: A = {
    val res = nextReservedToken
    nextReservedToken = t.next(nextReservedToken)
    res
  }

  def max = nextReservedToken

  def tokenize(in: String) = split(in).map(s => data.known.getOrElse(s, t.unknown))
}

object WordTokenizer {
  private val pattern = """'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""".r

  def split(in: String): Seq[String] = {
    var res = Vector.empty[String]
    var s = in
    while (s.length > 0) {
      pattern.findFirstMatchIn(s).map { m =>
        if (m.start > 0) {
          res :+= s.substring(0, m.start)
        }
        if (m.start != m.end) {
          res :+= s.substring(m.start, m.end)
        }
        s = s.substring(m.end)
      }.getOrElse {
        // No more matches, take rest
        res :+= s
        s = ""
      }
    }
    res
  }

  def train[A: Token](data: Iterable[String], minCount: Int = 1): WordData[A] = {
    val t = summon[Token[A]]

    case class Entry(id: A, count: Int) {
      def again = new Entry(id, count + 1)
    }

    var known = Map.empty[String, Entry]
    var next = t.unknown

    data.foreach { in =>
      for (s <- split(in)) {
        known.get(s).map { entry =>
          known += s -> entry.again
        }.getOrElse {
          next = t.next(next) // 0 is reserved for unknown
          known += s -> Entry(next, 1)
        }
      }
    }

    WordData(known.view.filter(_._2.count >= minCount).mapValues(_.id).toMap)
  }
}
