import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ResidualConvUnit(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)

class FusionBlock(nn.Module):
    def __init__(self, c, extra_block=None):
        super().__init__()
        self.rcu_skip = ResidualConvUnit(c)
        self.rcu_main = ResidualConvUnit(c)
        self.proj = nn.Conv2d(c, c, kernel_size=1)
        self.extra = extra_block(c) if extra_block is not None else nn.Identity()

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            x = x + self.rcu_skip(skip)
        x = self.rcu_main(x)
        return self.extra(self.proj(x))

class DPTDecoder(nn.Module):
    def __init__(self, vit_dim=768, chans=(96, 192, 384, 768), dec_ch=256, extra_block=None):
        super().__init__()
        self.project = nn.ModuleList([nn.Conv2d(vit_dim, c, kernel_size=1) for c in chans])
        self.resample = nn.ModuleList([
            nn.ConvTranspose2d(chans[0], chans[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(chans[1], chans[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(chans[3], chans[3], kernel_size=3, stride=2, padding=1)
        ])
        self.to_dec = nn.ModuleList([nn.Conv2d(c, dec_ch, kernel_size=3, padding=1, bias=False) for c in chans])
        self.fuse = nn.ModuleList([FusionBlock(dec_ch, extra_block) for _ in range(4)])

    def forward(self, feats):
        xs = [self.to_dec[i](self.resample[i](self.project[i](f))) for i, f in enumerate(feats)]
        x = self.fuse[3](xs[3])
        x = self.fuse[2](x, skip=xs[2])
        x = self.fuse[1](x, skip=xs[1])
        x = self.fuse[0](x, skip=xs[0])
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

class DesmokeHead(nn.Module):
    def __init__(self, c=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(c // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1),
        )
        self.smoke = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, feat, img):
        o = self.net(feat)
        K, B = o[:, :1], o[:, 1:]
        J = K * img - B + img
        denom = F.softplus(K + 1.0) + 1e-4
        airlight = B / denom
        S = self.smoke(airlight)
        return J, S

class SurgicalDesmoker(nn.Module):
    def __init__(self, img_size=(512, 640), pretrained=True, mae_ckpt=None, extra_block=None, dec_ch=256):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224.mae",
            pretrained=pretrained,
            img_size=img_size,
            dynamic_img_size=True,
            num_classes=0
        )
        if mae_ckpt:
            ckpt = torch.load(mae_ckpt, map_location="cpu")
            self.vit.load_state_dict(ckpt.get("model", ckpt), strict=False)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.blocks = [2, 5, 8, 11]
        self.decoder = DPTDecoder(self.vit.embed_dim, dec_ch=dec_ch, extra_block=extra_block)
        self.head = DesmokeHead(dec_ch)

    def forward(self, img):
        x = (img - self.mean) / self.std
        feats = self.vit.get_intermediate_layers(x, n=self.blocks, reshape=True)
        decoded = self.decoder(list(feats))
        return self.head(decoded, img)

    @torch.no_grad()
    def predict(self, img):
        J, S = self.forward(img)
        return J.clamp(0.0, 1.0), S.clamp(0.0, 1.0)
