import torch
import torch.nn as nn
import torch.nn.functional as F

class GPUSmokeSynthesizer(nn.Module):
    def __init__(self, octaves=5, persistence=0.55):
        super().__init__()
        self.octaves = octaves
        self.persistence = persistence

    @torch.no_grad()
    def forward(self, clean):
        b, _, h, w = clean.shape
        device, dtype = clean.device, clean.dtype

        out = torch.zeros(b, 1, h, w, device=device, dtype=dtype)
        amp, tot = 1.0, 0.0
        for o in range(self.octaves):
            gh = max(2, int(h / (2 ** (self.octaves - o - 1))))
            gw = max(2, int(w / (2 ** (self.octaves - o - 1))))
            n = torch.rand(b, 1, gh, gw, device=device, dtype=dtype)
            out += amp * F.interpolate(n, size=(h, w), mode="bilinear", align_corners=False)
            tot += amp
            amp *= self.persistence

        out = out / max(tot, 1e-6)
        min_v = out.amin(dim=(-2, -1), keepdim=True)
        max_v = out.amax(dim=(-2, -1), keepdim=True)
        noise = (out - min_v) / (max_v - min_v + 1e-6)

        thr = torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 0.35 + 0.20
        gamma = torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 0.90 + 0.70
        density = torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 0.65 + 0.35

        alpha = (((noise - thr) / (1.0 - thr).clamp(min=1e-4)).clamp(0.0, 1.0) ** gamma) * density
        alpha = F.avg_pool2d(alpha, kernel_size=9, stride=1, padding=4)

        gray = torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 0.33 + 0.65
        color_offset = 1.0 + 0.05 * torch.randn(b, 3, 1, 1, device=device, dtype=dtype)
        smoke_color = (gray.expand(-1, 3, -1, -1) * color_offset).clamp(0.0, 1.0)

        smoky = (clean * (1.0 - alpha) + smoke_color * alpha).clamp(0.0, 1.0)
        return smoky, alpha
