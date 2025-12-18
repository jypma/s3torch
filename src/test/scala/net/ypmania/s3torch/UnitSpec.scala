package net.ypmania.s3torch

import org.scalatest.funspec.AnyFunSpec

import org.bytedeco.pytorch
import org.scalactic.Equality

abstract class UnitSpec extends AnyFunSpec {
  def withSeed[T](seed: Long)(fn: => T): T = {
    UnitSpec.synchronized {
      pytorch.global.torch.manual_seed(seed)
      fn
    }
  }
  implicit val doubleEqual:Equality[Double] = new Equality[Double] {
    def areEqual(a: Double, b: Any): Boolean = b match {
      case d: Float if Math.abs(d - a) < 0.0001 => true
      case d: Double if Math.abs(d - a) < 0.0001 => true
      case _ => false
    }
  }
  implicit val floatEqual:Equality[Float] = new Equality[Float] {
    def areEqual(a: Float, b: Any): Boolean = b match {
      case d: Float if Math.abs(d - a) < 0.0001 => true
      case d: Double if Math.abs(d - a) < 0.0001 => true
      case _ => false
    }
  }
  implicit val float1dEqual:Equality[Seq[Float]] = new Equality[Seq[Float]] {
    def areEqual(a: Seq[Float], b: Any): Boolean = b match {
      case seq:Seq[?] => a.zip(seq).forall(floatEqual.areEqual(_,_))
      case _ => false
    }
  }
  implicit val double1dEqual:Equality[Seq[Double]] = new Equality[Seq[Double]] {
    def areEqual(a: Seq[Double], b: Any): Boolean = b match {
      case seq:Seq[?] => a.zip(seq).forall(doubleEqual.areEqual(_,_))
      case _ => false
    }
  }
  implicit val float2dEqual:Equality[Seq[Seq[Float]]] = new Equality[Seq[Seq[Float]]] {
    def areEqual(a: Seq[Seq[Float]], b: Any): Boolean = b match {
      case seq:Seq[?] => a.zip(seq).forall(float1dEqual.areEqual(_,_))
      case _ => false
    }
  }
  implicit val double2dEqual:Equality[Seq[Seq[Double]]] = new Equality[Seq[Seq[Double]]] {
    def areEqual(a: Seq[Seq[Double]], b: Any): Boolean = b match {
      case seq:Seq[?] => a.zip(seq).forall(double1dEqual.areEqual(_,_))
      case _ => false
    }
  }
}

object UnitSpec
