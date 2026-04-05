package net.ypmania.s3torch.tokenizer

import net.ypmania.s3torch
import net.ypmania.s3torch.token.Token

case class WordData[T](known: Map[String, T], inverse: Map[T, String])
object WordData {
  def apply[T](known: Map[String, T]): WordData[T] = WordData(known, known.map(_.swap))
}

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

  override def tokenize(in: String) = split(in).map(s => data.known.getOrElse(s, t.unknown))

  override def untokenize(seq: Seq[A]): String = {
    seq.map(t => data.inverse.getOrElse(t, "�")).mkString
  }
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
