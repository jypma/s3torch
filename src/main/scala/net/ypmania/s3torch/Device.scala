package net.ypmania.s3torch

import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch

sealed abstract class DeviceType(val native: torch.DeviceType) {
  DeviceType.fromNative = DeviceType.fromNative + (native.value -> this)
}

object DeviceType {
  private var fromNative = Map.empty[Byte, DeviceType]

  def of(native: torch.DeviceType): DeviceType = {
    fromNative(native.value)
  }

  case object CPU extends DeviceType(torch.DeviceType.CPU)
  case object CUDA extends DeviceType(torch.DeviceType.CUDA)
}

abstract class Device(val deviceType: DeviceType, val index: Byte = -1) {
  def native: pytorch.Device = pytorch.Device(deviceType.native.value, index)
}

object Device {
  /** The CPU device */
  case object CPU extends Device(DeviceType.CPU)
  /** The first CUDA device found */
  case object CUDA extends Device(DeviceType.CUDA)
}
