package net.ypmania.s3torch.nn

import org.scalatest.Assertions._

import net.ypmania.s3torch.*

import Dim.Static
import Dim.Dynamic
import scala.reflect.ClassTag
import net.ypmania.s3torch.internal.Torch
import java.nio.file.Files
import java.nio.file.Path
import scala.util.Using
import java.io.FileOutputStream

class ModuleSpec extends UnitSpec {
  case object ExampleDim extends Dim.Static[3L]

  class ExampleModule extends Module {
    val dropout = addModule("dropout", Dropout())
    val buffer = addBuffer("buffer", Tensor.zeros(ExampleDim))
    val param = addParameter("param", Tensor.zeros(ExampleDim))
  }

  describe("Module") {
    it("should save itself to a file and then load again") {
      val mod = new ExampleModule
      mod.buffer(1) = 1
      mod.save("/tmp/test.pt")

      val mod2 = new ExampleModule().load("/tmp/test.pt")
      assert(mod2.buffer.value.toSeq == Seq(0, 1, 0))
    }

    it("should save itself to an array and then load again") {
      val mod = new ExampleModule
      mod.buffer(1) = 1
      val saved = mod.save

      val mod2 = new ExampleModule().load(saved)
      assert(mod2.buffer.value.toSeq == Seq(0, 1, 0))
    }

    it("should save itself to a file and load that from an array") {
      val mod = new ExampleModule
      mod.buffer(1) = 1
      mod.save("/tmp/test2.pt")
      val saved = Files.readAllBytes(Path.of("/tmp/test2.pt"));

      val mod2 = new ExampleModule().load(saved)
      assert(mod2.buffer.value.toSeq == Seq(0, 1, 0))
    }
  }
}
