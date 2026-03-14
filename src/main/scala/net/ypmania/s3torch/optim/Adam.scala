package net.ypmania.s3torch.optim

import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Tensor
import org.bytedeco.pytorch
import scala.util.Using

class Adam(native: pytorch.Adam) {
  /** Perform a single optimization step. */
  def step(): Unit = native.step()

  /** Reset the gradients of all optimized tensors.
    * @param setToNone Instead of setting to zero, set the grads to None.
    */
  def zeroGrad(setToNone: Boolean = false): Unit = native.zero_grad(setToNone)

  def load(filename: String): this.type = {
    Using(new pytorch.InputArchive) { archive =>
      archive.load_from(filename)
      native.load(archive)
    }
    this
  }

  def save(filename: String): Unit = {
    Using(new pytorch.OutputArchive) { archive =>
      native.save(archive)
      archive.save_to(filename)
    }
  }
}

object Adam {
  def apply[D <: Device](
    parameters: Iterable[Tensor[?, ?, D]],
    learningRate: Double = 0.001,
    eps: Double = 1e-8,
    beta1: Double = 0.9,
    beta2: Double = 0.999,
    weightDecay: Double = 0,
    amsGrad: Boolean = false
  ): Adam = {
    val vect = new pytorch.TensorVector
    parameters.foreach(p => vect.push_back(p.native))
    val opts = new pytorch.AdamOptions
    opts.set_lr(learningRate)
    opts.eps().put(eps)
    opts.betas().put(0, beta1)
    opts.betas().put(1, beta2)
    opts.weight_decay().put(weightDecay)
    opts.amsgrad().put(amsGrad)
    new Adam(new pytorch.Adam(vect, opts))
  }
}
