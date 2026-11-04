import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Surgical Desmoker Training and Evaluation")
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--dec_ch", type=int, default=256)
    parser.add_argument("--mae_ckpt", type=str, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--clean_dir", type=str, default="./data/clean_frames")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./trained_models")
    parser.add_argument("--ckpt", type=str, default="./trained_models/latest.pt")
    parser.add_argument("--test_image", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="./results")
    return parser
