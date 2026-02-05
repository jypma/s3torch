package net.ypmania.s3torch.nn

import scala.util.Using

import net.ypmania.s3torch.Tensor
import net.ypmania.s3torch.DType

import org.bytedeco.pytorch
import net.ypmania.s3torch.Shape
import org.bytedeco.javacpp.Pointer
import java.nio.ByteBuffer
import net.ypmania.s3torch.Default
import org.bytedeco.pytorch
import net.ypmania.s3torch.Device

/** The base class for all nn modules */
abstract class AbstractModule[D <: Device, T <: DType](private[AbstractModule] val native: pytorch.Module) {
  type This[D <: Device, T <: DType]

  /** Registers the given module as a sub-module (so its state is loaded/saved together), and returns it. */
  protected def addModule[M <: AbstractModule[D, T]](name: String, child: M): M = {
    native.register_module(name, child.native)
    child
  }

  /** Registers the given modules as a sub-module list (so their state is loaded/saved together), and returns the same list. */
  protected def addModules[M <: AbstractModule[D, T]](name: String, children: Seq[M]): Seq[M] = {
    val list = new pytorch.ModuleListImpl
    for (child <- children) {
      list.push_back(child.native)
    }
    native.register_module(name, list)
    children
  }

  /** Registers the given buffer (so its state is loaded/saved together), and returns it. */
  protected def addBuffer[S <: Shape](name: String, buffer: Tensor[S, T, D]): Tensor[S, T, D] = {
    native.register_buffer(name, buffer.native)
    buffer
  }

  /** Registers the given parameter (so its state is loaded/saved together), and returns it. */
  protected def addParameter[S <: Shape](name: String, parameter: Tensor[S, T, D]): Tensor[S, T, D] = {
    native.register_parameter(name, parameter.native)
    parameter
  }

  /** Converts all sub-modules, parameters and buffers to the given target DType. This is a mutable operation, so only the
    * returned type and instance should be used. The source object (and its type )is no longer valid after this operation.
    * Since most module calculate gradients on their content, the target DType must be "Floaty", i.e. float or complex. */
  def to[T1 <: DType.Floaty](dtype: T1): This[D, T1] = {
    native.to(dtype.native, false)
    this.asInstanceOf[This[D, T1]]
  }

  /** Converts all sub-modules, parameters and buffers to the given target Device. This is a mutable operation, so only the
    * returned type and instance should be used. The source object (and its type )is no longer valid after this operation. */
  def to[D1 <: Device](device: D1): This[D1, T] = {
    native.to(device.native, false)
    this.asInstanceOf[This[D1, T]]
  }

  /** Converts all sub-modules, parameters and buffers to the given target Device and DType. This is a mutable operation, so only the
    * returned type and instance should be used. The source object (and its type )is no longer valid after this operation.
    * Since most module calculate gradients on their content, the target DType must be "Floaty", i.e. float or complex. */
  def to[D1 <: Device, T1 <: DType.Floaty](device: D1, dtype: T1): This[D1, T1] = {
    native.to(device.native, dtype.native, false)
    this.asInstanceOf[This[D1, T1]]
  }

  /** Alias for .to(dtype) where the dtype of type T is available as a given of Default[T]. */
  def toDType[T1 <: DType.Floaty](using t:Default[T1]) = to(t.value)

  /** Alias for .to(device) where the device of type D is available as a given of Default[D]. */
  def toDevice[D1 <: Device](using d:Default[D]) = to(d.value)

  /** Alias for .to(device, dtype) where the dtype of type T, and the device of type D, are available as a given of Default[T] and Default[D]. */
  def toDeviceDType[D <: Device, T1 <: DType.Floaty](using d: Default[D], t:Default[T1]) = to(d.value, t.value)

  /** Loads from the given file in pytorch "pt" format */
  def load(filename: String): this.type = {
    Using(new pytorch.InputArchive) { archive =>
      archive.load_from(filename)
      native.load(archive)
    }
    this
  }

  /** Loads pytorch "pt" formst, by repeatedly calling [read] with a target byte buffer to fill, and how many bytes to set there for that call. */
  def load(size: Long, read: (ByteBuffer, Long) => Unit): this.type = {
    Using(new pytorch.InputArchive) { archive =>
      // for pytorch 2.7.1+ : Reader() and SizeTSupplier() have moved from pytorch.functions to pytorch.
      archive.load_from(new pytorch.functions.Reader() {
        override def call(pos: Long, buf: Pointer, nbytes: Long): Long = {
          buf.limit(nbytes)
          read(buf.asByteBuffer, pos)
          nbytes
        }
      }, new pytorch.functions.SizeTSupplier() {
        override def call = size
      })
      native.load(archive)
    }
    this
  }

  /** Loads from the given content in pytorch "pt" format */
  def load(content: Array[Byte]): this.type = {
    load(content.size, (buf, pos) => buf.put(content, pos.toInt, buf.limit()))
    this
  }

  /** Saves as pytorch "pt" format, into a ZIP archive with the zip base filename as root directory. [fn] is repeatedly called, until the module is fully saved. */
  def save(filename: String): Unit = {
    Using(new pytorch.OutputArchive) { archive =>
      native.save(archive)
      archive.save_to(filename)
    }
  }

  /** Saves as pytorch "pt" format, into a ZIP archive with "archive" as root. */
  def save: Array[Byte] = {
    // TODO there's probably a more performant way to write this.
    var res: Seq[Array[Byte]] = Vector.empty
    save { buf =>
      val a = new Array[Byte](buf.limit())
      buf.get(a)
      res :+= a
    }
    res.flatten.toArray
  }

  /** Saves as pytorch "pt" format, into a ZIP archive with "archive" as root. [fn] is repeatedly called, until the module is fully saved. */
  def save(fn: ByteBuffer => Unit) = {
    Using(new pytorch.OutputArchive) { archive =>
      native.save(archive)
      // for pytorch 2.7.1+ : ArchiveWriter() has moved from pytorch.functions to pytorch.
      archive.save_to(new pytorch.functions.ArchiveWriter() {
        override def call(buf: Pointer, nbytes: Long) = {
          buf.limit(nbytes)
          fn(buf.asByteBuffer)
          nbytes
        }
      })
    }
  }
}

object AbstractModule {
  /** The default DType for which all pytorch modules are created */
  type CreationDType = DType.Float32.type
}

/** The base class for user-defined nn modules */
abstract class Module[D <: Device, T <: DType](using Default[D], Default[T]) extends AbstractModule[D, T](new pytorch.Module()) {

}

object Module {
}
