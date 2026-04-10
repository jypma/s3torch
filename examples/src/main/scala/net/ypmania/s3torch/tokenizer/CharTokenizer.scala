package net.ypmania.s3torch.tokenizer

import net.ypmania.s3torch
import net.ypmania.s3torch.token.Token

case class CharData[T](known: Map[Char, T], inverse: Map[T, Char])
object CharData {
  def apply[T](known: Map[Char, T]): CharData[T] = CharData(known, known.map(_.swap))
}

class CharTokenizer[A: Token](data: CharData[A]) extends Tokenizer[A] {
  private val t = summon[Token[A]]
  private var nextReservedToken = t.max(data.known.values)

  protected def reservedToken: A = {
    val res = nextReservedToken
    nextReservedToken = t.next(nextReservedToken)
    res
  }

  override def max = nextReservedToken

  override def tokenize(in: String) = in.map(ch => data.known.getOrElse(ch, t.unknown))

  override def untokenize(seq: Seq[A]) = seq.map(t => data.inverse.getOrElse(t, "�")).mkString
}

object CharTokenizer {
  def train[A: Token](data: Iterable[String], minCount: Int = 1): CharData[A] = {
    val t = summon[Token[A]]

    class Entry(val id: A, var count: Int) {
      def again: Unit = count += 1
    }

    var known = Map.empty[Char, Entry]
    var next = t.unknown

    data.foreach { in =>
      for (ch <- in) {
        known.get(ch).map { entry =>
          entry.again
        }.getOrElse {
          next = t.next(next) // 0 is reserved for unknown
          known += ch -> Entry(next, 1)
        }
      }
    }

    CharData(known.view.filter(_._2.count >= minCount).mapValues(_.id).toMap)
  }
}
