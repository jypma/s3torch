package net.ypmania.s3torch.tokenizer

import net.ypmania.s3torch.token.Token

abstract class Tokenizer[A: Token] {
  def max: A
  def tokenize(in: String): Seq[A]
  def untokenize(seq: Seq[A]): String
}
