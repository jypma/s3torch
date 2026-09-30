package net.ypmania.s3torch

/** The known size of a tensor with shape S */
case class Size[S <: Tuple] private[s3torch] (size: Seq[Long]) {

}
