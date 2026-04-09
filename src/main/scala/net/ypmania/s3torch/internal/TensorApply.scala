package net.ypmania.s3torch.internal

import net.ypmania.s3torch.DType
import net.ypmania.s3torch.DTypeOps
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.DeviceType
import net.ypmania.s3torch.Dim
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch

import compiletime.ops.int.ToLong

trait TensorApply[V] {
  type OutShape <: Tuple
  def apply[D <: Device](value: V, device: D): pytorch.Tensor
}

object TensorApply {
  /** Calculates the base type V of a Seq[Seq[...[V]]], or a Tuple(V, V, V) */
  type BaseType[V] = V match {
    case Seq[elem] => BaseType[elem]
    case Tuple => BaseType[Tuple.Union[V]]
    case _ => V
  }

  /** Turns a Seq[Seq[...[V]]], or a Tuple(V, V, V), into an appropriate tuple for a Tensor's shape (with Dim types as dimensions) */
  type ToShape[V] <: Tuple = V match {
    case Seq[elem] => Dim.Dynamic *: ToShape[elem]
    case Tuple => Dim.Static[ToLong[Tuple.Size[V]]] *: ToShape[Tuple.Union[V]]
    case _ => EmptyTuple
  }

  abstract class Primitive[V](dType: DType, ops:DTypeOps[?, V]) extends TensorApply[V] {
    type OutShape = EmptyTuple

    def apply[D <: Device](value: V, device: D) = {
      torch.scalar_tensor(
        ops.fromScalar(value),
        Torch.tensorOptions(dType, device)
      )
    }
  }
  abstract class Seq1D[S, V](dType: DType, ops: DTypeOps[?, V])(using toSeq: ToSeq[S, V]) extends TensorApply[S] {
    type OutShape = ToShape[S]

    def apply[D <: Device](value: S, device: D) = {
      val seq = toSeq(value)

      device.deviceType match {
        case DeviceType.CPU =>
          val ptr = ops.toPointer(seq)
          torch
            .from_blob(ptr._1, Array(seq.length.toLong), Torch.tensorOptions(dType, device))
            .clone() // from_blob, if running on CPU, retains a reference to the original ByteBuffer, which might be GC'ed.
        case _ =>
          val opts = Torch.tensorOptions(dType, Device.CPU)
          val ptr = ops.toPointer(seq)
          torch
            .from_blob(ptr._1, Array(seq.length.toLong), opts)
            .to(device.native, opts.dtype())
      }
    }
  }
  abstract class Seq2D[S1, S2, V](dType: DType, ops: DTypeOps[?, V])(using toSeq1: ToSeq[S1, S2], toSeq2: ToSeq[S2, V], toTensor: TensorApply[Seq[V]]) extends TensorApply[S1] {
    type OutShape = ToShape[S1]

    def apply[D <: Device](value: S1, device: D) = {
      val seqs1 = toSeq1(value)
      val seq = seqs1.map(s => toSeq2(s)).flatten
      toTensor(seq, device).view(seqs1.size, seq.length / seqs1.size)
    }
  }

  abstract class Seq3D[S1, S2, S3, V](dType: DType, ops: DTypeOps[?, V])(using toSeq1: ToSeq[S1, S2], toSeq2: ToSeq[S2, S3], toSeq3: ToSeq[S3, V], toTensor: TensorApply[Seq[V]]) extends TensorApply[S1] {
    type OutShape = ToShape[S1]

    def apply[D <: Device](value: S1, device: D) = {
      val seq = toSeq1(value).map(s2 => toSeq2(s2).map(s3 => toSeq3(s3)))
      toTensor(seq.flatten.flatten, device).view(seq.size, seq.head.size, seq.head.head.size)
    }
  }

  given [V](using t:DTypeFor[V], ops: DTypeOps[t.Out, V]): Primitive[V](t.dType, ops)
  given [S, V](using t:DTypeFor[BaseType[S]], toSeq: ToSeq[S, V], ops: DTypeOps[t.Out, V]): Seq1D[S, V](t.dType, ops)
  given [S1, S2, V](using t:DTypeFor[BaseType[S1]], ops: DTypeOps[t.Out, V])(using ToSeq[S1, S2], ToSeq[S2, V], TensorApply[Seq[V]]): Seq2D[S1, S2, V](t.dType, ops)
  given [S1, S2, S3, V](using t:DTypeFor[BaseType[S1]], ops: DTypeOps[t.Out, V])(using ToSeq[S1, S2], ToSeq[S2, S3], ToSeq[S3, V], TensorApply[Seq[V]]): Seq3D[S1, S2, S3, V](t.dType, ops)
}
