"""Hybrid residue estimate on the synthetic dev families: CNN below a cutoff, Lab projection above.

Two gates are compared: the projection's own estimate (available on the line without the CNN) and
the CNN's estimate. When the gate reads at or above CUTOFF the projection is reported, otherwise
the CNN. The cutoffs are fixed in advance
at bin edges (0.3, where the v3 heavy arm starts, and 0.5, above which a vial is mostly the
previous recipe), not swept. The checkpoints' epochs were chosen on the same dev families, so the
CNN numbers are slightly optimistic. No real image or Run 5 value is read.

    python eval_carryover_hybrid.py RENDERS... --models DIR... --out hybrid.json [--cutoffs 0.3 0.5]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from replay.carryover_synthetic import BINS, changeovers, measure  # noqa: E402
from train_carryover_estimator import Net, load  # noqa: E402


def by_bin(truth, estimates):
    out = []
    for low, high in BINS:
        sel = truth == 0 if high == 0 else (truth > low) & (truth <= high)
        if sel.any():
            out.append(dict(f_range=[low, high], n=int(sel.sum()),
                            **{name: float(np.abs(e[sel] - truth[sel]).mean()) for name, e in estimates.items()}))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("renders", type=Path, nargs="+")
    parser.add_argument("--models", type=Path, nargs="+", required=True, help="training output folders with best.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[0.3, 0.5])
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    truth = {c["pair"]: c for c in changeovers([row for batch in args.renders for row in measure(batch)])}
    dev = [(p, x) for p, x in load(args.renders)["dev"] if p in truth]
    x = torch.tensor(np.stack([x for _, x in dev])).to(device)
    f = np.array([truth[p]["f"] for p, _ in dev])
    projection = np.array([truth[p]["residue_estimate"] for p, _ in dev])

    results = []
    for folder in args.models:
        model = Net().to(device)
        model.load_state_dict(torch.load(folder / "best.pt", map_location=device))
        model.eval()
        with torch.no_grad():
            cnn = model(x).cpu().numpy()
        estimates = dict(projection=projection, cnn=cnn,
                         **{f"{gate}_gate_{c}": np.where(g >= c, projection, cnn)
                            for gate, g in (("projection", projection), ("cnn", cnn)) for c in args.cutoffs})
        results.append(dict(model=folder.name, mae={k: float(np.abs(e - f).mean()) for k, e in estimates.items()},
                            by_bin=by_bin(f, estimates),
                            switched_to_projection={f"{gate}_gate_{c}": int((g >= c).sum()) for c in args.cutoffs
                                                    for gate, g in (("projection", projection), ("cnn", cnn))}))
        print(folder.name, {k: round(v, 4) for k, v in results[-1]["mae"].items()}, flush=True)
    args.out.write_text(json.dumps(dict(dev_changeovers=len(dev), cutoffs=args.cutoffs, gates=["projection estimate", "cnn estimate"],
                                        cutoff_choice="fixed bin edges, not swept", real_data_inputs=[],
                                        results=results), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
