"""BC pretrain for the transformer (RoPE) student on the merged BC dataset.

The RoPEFrameEncoder is cold-started (random init) while readout/heads/value
warm-start from the OmniStudent BC checkpoint via RLStudent.load_bc's
shape filter. Training signal = the same action losses as the BC pass in
student_rl.py (move MSE + aim MSE + shoot/super CE) plus the next-feature
prediction auxiliary loss (forward_full, weight --aux-fp).

Frames are encoded ONCE through the frozen DINO backbone into a feature
table (cached on disk); windows then index into the table, so training is
pure transformer-side and fast. Windows respect episode boundaries via
student_vision.make_windows (front-padded at episode starts).

Usage:
    .venv/Scripts/python student_tx_pretrain.py --epochs 10
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from student_vision import make_windows, two_scale_tokens
from student_rl import RLStudent, N_FRAMES, STRIDE, N_COARSE, HIST_LEN, INIT

OUT = os.path.join("runs", "student_tx")
DATA = os.path.join("runs", "student_vision", "bc_data_merged.npz")
ACTS = os.path.join("runs", "student_vision", "bc_acts_merged.npz")
FEATS_CACHE = os.path.join(OUT, "bc_feats_merged.pt")


def encode_all(net: RLStudent, frames: np.ndarray, device: str,
               batch: int = 512, use_cache: bool = True) -> torch.Tensor:
    """(N, 90, 126, 3) uint8 -> (N, 384) float32 feature table on device.

    Full-dataset encodes are cached on disk; a .part checkpoint is written
    every 40 batches so a killed run resumes instead of starting over."""
    part = FEATS_CACHE + ".part"
    if use_cache and os.path.exists(FEATS_CACHE):
        print(f"loading cached features from {FEATS_CACHE}")
        return torch.load(FEATS_CACHE, map_location=device)
    feats = []
    start = 0
    if use_cache and os.path.exists(part):
        d = torch.load(part, map_location=device)
        feats, start = [d["feats"].to(device)], d["done"]
        print(f"resuming feature encode at {start}/{len(frames)}")
    t0 = time.perf_counter()
    with torch.no_grad():
        for nb, b in enumerate(range(start, len(frames), batch)):
            x = torch.from_numpy(frames[b:b + batch]).permute(0, 3, 1, 2)
            feats.append(net.encode(x.to(device)))
            done = min(b + batch, len(frames))
            if (nb + 1) % 20 == 0:
                print(f"  encoded {done}/{len(frames)} "
                      f"({(done - start) / (time.perf_counter() - t0):.0f}/s)",
                      flush=True)
            if use_cache and (nb + 1) % 40 == 0:
                os.makedirs(OUT, exist_ok=True)
                torch.save({"feats": torch.cat(feats).cpu(), "done": done},
                           part)
    feats = torch.cat(feats)
    if use_cache:
        os.makedirs(OUT, exist_ok=True)
        torch.save(feats.cpu(), FEATS_CACHE)
        if os.path.exists(part):
            os.remove(part)
    return feats


def encode_all(net: RLStudent, frames: np.ndarray, device: str,
               batch: int = 512, use_cache: bool = True,
               cache_path: str = FEATS_CACHE) -> torch.Tensor:
    """(N, 90, 126, 3) uint8 -> (N, 384) float32 feature table on device.

    Full-dataset encodes are cached on disk; a .part checkpoint is written
    every 40 batches so a killed run resumes instead of starting over."""
    part = cache_path + ".part"
    if use_cache and os.path.exists(cache_path):
        print(f"loading cached features from {cache_path}")
        return torch.load(cache_path, map_location=device)
    feats = []
    start = 0
    if use_cache and os.path.exists(part):
        d = torch.load(part, map_location=device)
        feats, start = [d["feats"].to(device)], d["done"]
        print(f"resuming feature encode at {start}/{len(frames)}")
    t0 = time.perf_counter()
    with torch.no_grad():
        for nb, b in enumerate(range(start, len(frames), batch)):
            x = torch.from_numpy(frames[b:b + batch]).permute(0, 3, 1, 2)
            feats.append(net.encode(x.to(device)))
            done = min(b + batch, len(frames))
            if (nb + 1) % 20 == 0:
                print(f"  encoded {done}/{len(frames)} "
                      f"({(done - start) / (time.perf_counter() - t0):.0f}/s)",
                      flush=True)
            if use_cache and (nb + 1) % 40 == 0:
                os.makedirs(OUT, exist_ok=True)
                torch.save({"feats": torch.cat(feats).cpu(), "done": done},
                           part)
    feats = torch.cat(feats)
    if use_cache:
        os.makedirs(OUT, exist_ok=True)
        torch.save(feats.cpu(), cache_path)
        if os.path.exists(part):
            os.remove(part)
    return feats


def pretrain(n_epochs: int, device: str, batch_size: int = 256,
             lr: float = 1e-3, aux_fp: float = 0.1, init: str = INIT,
             out: str = os.path.join(OUT, "bc_warm.pt"),
             limit_batches: int = 0, encode_batch: int = 512,
             limit_frames: int = 0, backbone: str = "dino",
             cnn_init: str | None = None, data: str = DATA,
             acts: str = ACTS):
    d = np.load(data)
    if "moves" in d:     # v5 single-file format (student_cnn_pretrain)
        frames = d["frames"][:limit_frames] if limit_frames else d["frames"]
        moves, aims, btns = d["moves"], d["aims"], d["btns"]
        ep_start = d["ep_start"][:len(frames)]
    else:                # legacy split pair (frames file + acts file)
        da = np.load(acts)
        frames = d["frames"][:limit_frames] if limit_frames else d["frames"]
        moves = da["moves"][:len(frames)]
        aims = da["aims"][:len(frames)]
        btns = da["btns"][:len(frames)]
        ep_start = d["ep_start"][:len(frames)]
    mv_all = torch.from_numpy(moves)
    am_all = torch.from_numpy(aims)
    bt_all = torch.from_numpy(btns)
    n = len(frames)
    ep_bounds = np.flatnonzero(ep_start)
    print(f"dataset: {n} frames, window={N_FRAMES}x stride={STRIDE} "
          f"+ {N_COARSE} pooled coarse blocks [tx two-scale, {backbone}]")

    net = RLStudent(encoder="tx", backbone=backbone).to(device)
    net.load_bc(init)
    if cnn_init:
        sd = torch.load(cnn_init, map_location="cpu")["cnn"]
        net.cnn.load_state_dict(sd)
        print(f"CNN encoder loaded from {cnn_init}")
    cache = FEATS_CACHE if backbone == "dino" else FEATS_CACHE.replace(
        ".pt", f"_{backbone}.pt")
    feats = encode_all(net, frames, device, encode_batch,
                       use_cache=not limit_frames, cache_path=cache)
    mv_all = mv_all.to(device)
    am_all = am_all.to(device)
    bt_all = bt_all.to(device)

    trainable = [p for n_, p in net.named_parameters()
                 if not n_.startswith(("backbone", "cnn"))]
    opt = torch.optim.Adam(trainable, lr=lr)
    n_val = max(1, n // 10)
    perm = np.random.RandomState(0).permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    steps_per_ep = (len(tr_idx) + batch_size - 1) // batch_size
    if limit_batches:
        steps_per_ep = min(steps_per_ep, limit_batches)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs * steps_per_ep, eta_min=lr / 10)

    def batch_loss(idx: np.ndarray):
        # two-scale window: stride-1 indices over the full retained history
        # (front-padded at episode starts), then fine stride + attention
        # pooling happen inside two_scale_tokens
        widx = torch.from_numpy(
            make_windows(idx, ep_bounds, HIST_LEN, 1)).to(device)
        idx_t = torch.from_numpy(idx).to(device)
        fw = two_scale_tokens(feats[widx], net.pool)   # (B, 158, 384)
        out, pred_seq = net.forward_full(fw)
        # pred_seq covers the 29 fine-token pairs; slice the target to match
        loss_fp = F.mse_loss(pred_seq,
                             fw[:, -pred_seq.shape[1]:].detach())
        loss = (F.mse_loss(out["move_mu"], mv_all[idx_t])
                + F.mse_loss(out["aim_mu"], am_all[idx_t])
                + F.cross_entropy(out["shoot"], bt_all[idx_t][:, 0])
                + F.cross_entropy(out["super"], bt_all[idx_t][:, 1])
                + aux_fp * loss_fp)
        return loss, loss_fp

    try:
        for ep in range(n_epochs):
            net.train()
            order = np.random.permutation(len(tr_idx))
            tot = tot_fp = 0.0
            nb = 0
            t0 = time.perf_counter()
            for b in range(0, len(order), batch_size):
                idx = tr_idx[order[b:b + batch_size]]
                loss, loss_fp = batch_loss(idx)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 0.5)
                opt.step()
                sched.step()
                tot += float(loss)
                tot_fp += float(loss_fp)
                nb += 1
                if nb % 200 == 0:
                    print(f"  ep {ep + 1} batch {nb} loss {tot / nb:.4f} "
                          f"fp {tot_fp / nb:.4f} "
                          f"({nb * batch_size / (time.perf_counter() - t0):.0f}"
                          f" samples/s)", flush=True)
                if limit_batches and nb >= limit_batches:
                    break
            # validation: same loss on the 10% holdout
            net.eval()
            vtot = vfp = 0.0
            vnb = 0
            with torch.no_grad():
                for b in range(0, len(val_idx), batch_size):
                    loss, loss_fp = batch_loss(val_idx[b:b + batch_size])
                    vtot += float(loss)
                    vfp += float(loss_fp)
                    vnb += 1
            print(f"[epoch {ep + 1}/{n_epochs}] loss {tot / max(nb, 1):.4f} "
                  f"fp {tot_fp / max(nb, 1):.4f} "
                  f"val loss {vtot / max(vnb, 1):.4f} "
                  f"val fp {vfp / max(vnb, 1):.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} "
                  f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            torch.save(net.state_dict(), out)
    except KeyboardInterrupt:
        print("interrupted; saving partial checkpoint")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(net.state_dict(), out)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aux-fp", type=float, default=0.1)
    ap.add_argument("--init", type=str, default=INIT)
    ap.add_argument("--out", type=str, default=os.path.join(OUT, "bc_warm.pt"))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit-batches", type=int, default=0,
                    help="cap batches per epoch (smoke tests)")
    ap.add_argument("--encode-batch", type=int, default=512,
                    help="batch size for the one-off feature encoding pass")
    ap.add_argument("--limit-frames", type=int, default=0,
                    help="use only the first N frames (smoke tests; the "
                         "feature cache is neither read nor written)")
    ap.add_argument("--backbone", choices=("dino", "cnn"), default="dino")
    ap.add_argument("--cnn-init", type=str, default=None,
                    help="stage-1 CNN checkpoint (student_cnn_pretrain --out)")
    ap.add_argument("--data", type=str, default=DATA)
    ap.add_argument("--acts", type=str, default=ACTS)
    args = ap.parse_args()
    pretrain(args.epochs, args.device, args.batch_size, args.lr,
             args.aux_fp, args.init, args.out, args.limit_batches,
             args.encode_batch, args.limit_frames, args.backbone,
             args.cnn_init, args.data, args.acts)


if __name__ == "__main__":
    main()
