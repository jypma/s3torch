package net.ypmania.s3torch.transformer

trait Tokenizer() {
  def tokenize(in: String): Seq[Int]
}

object Tokenizer {
  trait Trainer[T <: Tokenizer] {
    def train(in: String): Unit
    def complete: T
  }
}

case class WordTokenizer(known: Map[String, Int]) extends Tokenizer {
  import WordTokenizer._

  def tokenize(in: String) = split(in).map(s => known.getOrElse(s, 0))
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

  def train(in: Iterable[String], minCount: Int = 1): WordTokenizer = {
    val t = trainer(minCount)
    in.foreach(t.train)
    t.complete
  }

  case class Entry(id: Int, count: Int) {
    def again = new Entry(id, count + 1)
  }

  def trainer(minCount: Int = 1) = new Tokenizer.Trainer[WordTokenizer] {
    var known = Map.empty[String, Entry]

    def train(in: String): Unit = {
      for (s <- split(in)) {
        known.get(s).map { entry =>
          known += s -> entry.again
        }.getOrElse {
          val id = known.size + 1 // 0 is reserved for unknown
          known += s -> Entry(id, 1)
        }
      }
    }

    def complete = new WordTokenizer(known.view.filter(_._2.count >= minCount).mapValues(_.id).toMap)
  }
}
