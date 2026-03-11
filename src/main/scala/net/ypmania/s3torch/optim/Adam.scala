package net.ypmania.s3torch.optim

import net.ypmania.s3torch.Device
import net.ypmania.s3torch.Tensor
import org.bytedeco.pytorch

class Adam(native: pytorch.Adam) {
  def step(): Unit = native.step()

  def zeroGrad(): Unit = native.zero_grad()
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
