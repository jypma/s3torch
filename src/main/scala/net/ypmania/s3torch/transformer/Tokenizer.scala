package net.ypmania.s3torch.transformer

import net.ypmania.s3torch
import net.ypmania.s3torch.Dim
import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.Default
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.internal.FromScala
import net.ypmania.s3torch.internal.FromScala.ToScalar

/*
trait Token {
  case object DType extends s3torch.DType.Int32
  opaque type SType = Int
  def toInt(s: SType): Int = s
  def unknown: SType
  def next(t: SType): SType
  def max(ts: Iterable[SType]): SType

  given ToScalar[SType] = summon[ToScalar[Int]]
}
 */

trait Token[T] {
  //type DType <: s3torch.DType
  //def dType: DType
  def unknown: T
  def next(t: T): T
  def max(ts: Iterable[T]): T
  def toInt(t: T): Int
}

object Token {
  given Token[Int] = new IntToken {}
}

trait IntToken extends Token[Int] {
//  type DType = s3torch.DType.Int32
//  def dType = s3torch.DType.int32
  def unknown = 0
  def next(t: Int) = t + 1
  def max(ts: Iterable[Int]) = ts.max
  def toInt(t: Int) = t
}

trait TokenType {
  opaque type T = Int
  given Token[T] = new IntToken {}
  given ToScalar[T] = summon[ToScalar[Int]]
  abstract class DType extends s3torch.DType.Int32
  val dType = new DType {}
}

class Vectorizer[A: Token, T <: s3torch.DType](tk: Tokenizer[A], dtype: T) {
  private val t = summon[Token[A]]

  def toTensor[Dv <: Device](prefix: Seq[A], in: String, postfix: Seq[A])(using Default[Dv]): Tensor[Dim.Dynamic *: EmptyTuple, T, Dv] = {
    // TODO consider just .asInstanceOf, since we know t.toInt is just t
    Tensor((prefix ++ tk.tokenize(in) ++ postfix).map(t.toInt), dtype)
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
