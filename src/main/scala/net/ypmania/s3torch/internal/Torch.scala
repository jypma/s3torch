package net.ypmania.s3torch.internal

import net.ypmania.s3torch.DType
import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Layout
import org.bytedeco.pytorch
import org.bytedeco.pytorch._

object Torch {
  def manualSeed(seed: Long): Unit = pytorch.global.torch.manual_seed(seed)

  // TODO make CreationOptions case class, which can create itself from DefaultV2
  def tensorOptions(
      dtype: DType,
  ): pytorch.TensorOptions =
    pytorch
      .TensorOptions()
      .dtype(ScalarTypeOptional(dtype.native))
      .layout(LayoutOptional(Layout.Strided.toNative))
      .device(DeviceOptional(Device.CPU.native))
      .requires_grad(BoolOptional(false))
      .pinned_memory(BoolOptional(false))

  def tensorOptions(
    dtype: DType,
    device: Device
  ): pytorch.TensorOptions =
    pytorch
      .TensorOptions()
      .dtype(ScalarTypeOptional(dtype.native))
      .layout(LayoutOptional(Layout.Strided.toNative))
      .device(DeviceOptional(device.native))
      .requires_grad(BoolOptional(false))
      .pinned_memory(BoolOptional(false))
}
