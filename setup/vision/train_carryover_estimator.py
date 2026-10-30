"""Learned residue-fraction estimator on synthetic carryover renders, against the Lab projection.

Input per changeover: the liquid crops of the reference, previous and vial renders (renderer
liquid masks, 96x32 each, stacked to 9 channels). Target: the residue fraction f. Fitting uses
the train families; the epoch is chosen on the dev families (synthetic, held out as whole
families). The baseline is replay.carryover_synthetic's Lab projection on the same dev
changeovers. No real image or Run 5 value is read.

    python train_carryover_estimator.py RENDERS... --out DIR [--epochs 60] [--seed 29] [--balance]

--balance draws each epoch's train changeovers with equal weight per residue bin, so the few
heavy-residue (v3) changeovers are not drowned out by the clean and light ones.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from replay.carryover_synthetic import BINS, changeovers, measure  # noqa: E402

ROLES = ("reference", "previous", "vial")
CROP = (32, 96)  # width, height


def liquid_crop(batch, record):
    liquid = next(a for a in record["annotations"] if a["class_name"] == "liquid")
    if not liquid.get("bbox_xyxy"):
        return None
    rgb = np.asarray(Image.open(batch / record["image_file"]).convert("RGB"), dtype=np.float32) / 255
    mask = np.asarray(Image.open(batch / liquid["mask_file"])) > 0
    x0, y0, x1, y1 = liquid["bbox_xyxy"]
    crop = np.where(mask[..., None], rgb, 0)[y0:y1, x0:x1]
    return np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize(CROP, Image.BILINEAR),
                      dtype=np.float32) / 255


def load(batches):
    crops, meta = {}, {}
    for batch in batches:
        for line in (batch / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            p = record["parameters"]
            crop = liquid_crop(batch, record)
            if crop is not None:
                crops.setdefault(p["pair"], {})[p["role"]] = crop
                meta[p["pair"]] = (p["split"], p["family"])
    data = {"train": [], "dev": []}
    for pair, roles in sorted(crops.items()):
        if set(roles) == set(ROLES):
            split, _ = meta[pair]
            x = np.concatenate([roles[r] for r in ROLES], axis=2).transpose(2, 0, 1)
            data[split].append((pair, x))
    return data


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        block = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(9, 32), block(32, 64), block(64, 96), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("renders", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--balance", action="store_true", help="sample train changeovers equally per residue bin")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.time()

    rows = [row for batch in args.renders for row in measure(batch)]
    truth = {c["pair"]: c for c in changeovers(rows)}
    data = load(args.renders)
    data = {k: [(p, x) for p, x in v if p in truth] for k, v in data.items()}
    tensors = {k: (torch.tensor(np.stack([x for _, x in v])), torch.tensor([truth[p]["f"] for p, _ in v], dtype=torch.float32))
               for k, v in data.items() if v}
    if "train" not in tensors or "dev" not in tensors:
        parser.error("Need train and dev changeovers")
    projection_mae = float(np.mean([abs(truth[p]["residue_estimate"] - truth[p]["f"]) for p, _ in data["dev"]]))

    model = Net().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    xt, yt = tensors["train"]
    xd, yd = (t.to(device) for t in tensors["dev"])
    bins = torch.tensor([next(i for i, (low, high) in enumerate(BINS) if (f == 0 if high == 0 else low < f <= high))
                         for f in yt.tolist()])
    weights = 1 / torch.bincount(bins).float()[bins]
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.multinomial(weights, len(xt), replacement=True) if args.balance else torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            idx = order[i:i + 64]
            x, y = xt[idx].to(device), yt[idx].to(device)
            # Brightness jitter applied jointly keeps the three crops comparable, as one camera would.
            x = x * torch.empty(len(x), 1, 1, 1, device=device).uniform_(0.85, 1.15)
            loss = nn.functional.smooth_l1_loss(model(x), y, beta=0.02)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            mae = float((model(xd) - yd).abs().mean())
        history.append(dict(epoch=epoch, loss=float(loss), dev_mae=mae))
        if best is None or mae < best["dev_mae"]:
            best = dict(epoch=epoch, dev_mae=mae)
            torch.save(model.state_dict(), args.out / "best.pt")
        print(f"epoch {epoch} loss {float(loss):.4f} dev MAE {mae:.4f} (projection {projection_mae:.4f})", flush=True)

    # Dev error per residue bin at the selected epoch, next to the projection on the same changeovers.
    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        predicted = model(xd).cpu().numpy()
    truth_f = yd.cpu().numpy()
    projection = np.array([truth[p]["residue_estimate"] for p, _ in data["dev"]])
    by_bin = []
    for low, high in BINS:
        sel = truth_f == 0 if high == 0 else (truth_f > low) & (truth_f <= high)
        if sel.any():
            by_bin.append(dict(f_range=[low, high], n=int(sel.sum()),
                               cnn_mae=float(np.abs(predicted[sel] - truth_f[sel]).mean()),
                               projection_mae=float(np.abs(projection[sel] - truth_f[sel]).mean())))
    result = dict(train_changeovers=len(xt), dev_changeovers=len(xd), seed=args.seed, epochs=args.epochs,
                  sampling="equal per residue bin" if args.balance else "uniform",
                  device=torch.cuda.get_device_name() if device == "cuda" else "cpu",
                  best=best, projection_dev_mae=projection_mae, dev_by_bin=by_bin, seconds=time.time() - start,
                  selection="epoch with the lowest dev-family MAE (synthetic)", real_data_inputs=[],
                  history=history)
    (args.out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"best epoch {best['epoch']}: dev MAE {best['dev_mae']:.4f} vs projection {projection_mae:.4f}")


if __name__ == "__main__":
    main()
