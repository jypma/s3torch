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

/** The base class for all nn modules */
abstract class AbstractModule(private[AbstractModule] val native: pytorch.Module) {

  /** Registers the given module as a sub-module (so its state is loaded/saved together), and returns it. */
  protected def addModule[M <: AbstractModule](name: String, child: M): M = {
    native.register_module(name, child.native)
    child
  }

  /** Registers the given modules as a sub-module list (so their state is loaded/saved together), and returns the same list. */
  protected def addModules[M <: AbstractModule](name: String, children: Seq[M]): Seq[M] = {
    val list = new pytorch.ModuleListImpl
    for (child <- children) {
      list.push_back(child.native)
    }
    native.register_module(name, list)
    children
  }

  /** Registers the given buffer (so its state is loaded/saved together), and returns it. */
  protected def addBuffer[S <: Shape, T <: DType](name: String, buffer: Tensor[S, T]): Tensor[S, T] = {
    native.register_buffer(name, buffer.native)
    buffer
  }

  /** Registers the given parameter (so its state is loaded/saved together), and returns it. */
  protected def addParameter[S <: Shape, T <: DType](name: String, parameter: Tensor[S, T]): Tensor[S, T] = {
    native.register_parameter(name, parameter.native)
    parameter
  }

  private[nn] def to(dtype: DType): this.type = {
    native.to(dtype.scalarType, false)
    this
  }

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
      archive.load_from(new pytorch.Reader() {
        override def call(pos: Long, buf: Pointer, nbytes: Long): Long = {
          buf.limit(nbytes)
          read(buf.asByteBuffer, pos)
          nbytes
        }
      }, new pytorch.SizeTSupplier() {
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
      archive.save_to(new pytorch.ArchiveWriter() {
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
}

/** The base class for user-defined nn modules */
abstract class Module extends AbstractModule(new pytorch.Module()) {

}

object Module {
}
