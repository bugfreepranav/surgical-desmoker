from pathlib import Path
import numpy as np
import torch
from PIL import Image

from option import get_parser
from models.desmoker import SurgicalDesmoker

def run_test():
    opt = get_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(opt.out_dir).mkdir(parents=True, exist_ok=True)

    model = SurgicalDesmoker(img_size=(opt.img_h, opt.img_w), pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(opt.ckpt, map_location=device))

    with Image.open(opt.test_image) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        resized = img.resize((opt.img_w, opt.img_h), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(resized)).permute(2, 0, 1).float().div(255.0)[None].to(device)

    with torch.no_grad():
        J, S = model.predict(x)

    out_np = (J[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    smoke_np = (S[0, 0].cpu().numpy() * 255.0).astype(np.uint8)

    stem = Path(opt.test_image).stem
    Image.fromarray(out_np).resize((orig_w, orig_h), Image.Resampling.BILINEAR).save(f"{opt.out_dir}/{stem}_clean.png")
    Image.fromarray(smoke_np).resize((orig_w, orig_h), Image.Resampling.BILINEAR).save(f"{opt.out_dir}/{stem}_smoke.png")
    print(f"Saved results to {opt.out_dir}/")

if __name__ == "__main__":
    run_test()
