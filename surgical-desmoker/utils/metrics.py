import math
import torch.nn.functional as F

def calc_psnr(pred, target):
    mse = F.mse_loss(pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0)).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-10))
